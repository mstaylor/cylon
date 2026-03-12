##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

"""
Results aggregator for Cylon experiment summary files.

Parses summary CSVs (both old space-separated and new CSV formats),
computes per-run totals (sum of iterations) and cross-run mean/stddev,
outputs an aggregated CSV for chart generation.

Aggregation methodology (matching the reference spreadsheet):
  1. Each file = one run with N iterations (typically 10)
  2. Per-run: SUM timing columns across iterations → total run time (ms)
  3. Cross-run: MEAN and STD of the per-run totals → final values
  4. MS→S conversion applied after summing
"""

import logging
import re
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    LEGACY_COLUMNS, TIMING_COLUMNS, COST_COLUMNS,
    METRIC_COLUMNS,
)

logger = logging.getLogger(__name__)

# Columns that are in milliseconds and should be converted to seconds
# Timing columns recorded in milliseconds, converted to seconds in aggregated output.
# Note: lambda_duration_ms is an AWS billing metric and is NOT converted.
MS_TO_S_COLUMNS = [
    "avg_t", "elapsed_t", "max_t", "com_init_t", "barrier_t",
    "data_gen_t", "compute_t", "comm_t",
]


def extract_join_total(scaling_filepath: str) -> Optional[float]:
    """Extract join_total wall-clock time (seconds) from a scaling file.

    The scaling file contains a CSV line like:
      # csv,join_total_fmi-cylon_9100000_10,ok,438.22,438.22,...
    The 4th field (index 3) is the elapsed time in seconds.
    """
    if not os.path.isfile(scaling_filepath):
        return None
    try:
        with open(scaling_filepath, 'r') as f:
            for line in f:
                if line.startswith('# csv,join_total') or line.startswith('# csv,groupby_total'):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        return float(parts[3])
    except (ValueError, IOError):
        pass
    return None


def find_scaling_file(summary_filepath: str) -> Optional[str]:
    """Derive the scaling file path from a summary file path."""
    dirname = os.path.dirname(summary_filepath)
    basename = os.path.basename(summary_filepath)
    scaling_name = basename.replace('_summary_', '_scaling_')
    if scaling_name == basename:
        scaling_name = basename.replace('cylon_summary_', 'cylon_scaling_')
    scaling_path = os.path.join(dirname, scaling_name)
    if os.path.isfile(scaling_path):
        return scaling_path
    return None


def parse_node_count_from_filename(filename: str) -> Optional[int]:
    """Extract node count from filename like 'cylon_summary_test_..._4node.txt'."""
    match = re.search(r'_(\d+)node', filename)
    if match:
        return int(match.group(1))
    return None


def parse_summary_csv(filepath: str) -> pd.DataFrame:
    """Parse a summary file, auto-detecting format.

    Older experiments used '### ' prefixed space-separated values (7 columns).
    Newer experiments use standard CSV with a header row.

    Returns DataFrame with all available columns. Missing columns are NaN.
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()

    if first_line.startswith('##'):
        # Old space-separated format: "###  w 1 9100000 9100000 0 3000.76 10111764"
        rows = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r'^#+\s*', '', line)
                parts = line.split()
                if len(parts) >= 7:
                    rows.append({
                        'scaling': parts[0],
                        'world': int(parts[1]),
                        'rows': int(parts[2]),
                        'max_value': int(parts[3]),
                        'rank': int(parts[4]),
                        'avg_t': float(parts[5]),
                        'tot_l': int(parts[6]),
                    })
        df = pd.DataFrame(rows)
    else:
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.warning(f"Failed to parse {filepath}: {e}")
            return pd.DataFrame()

    if df.empty:
        return df

    # Add missing columns with NaN for backward compatibility
    all_columns = LEGACY_COLUMNS + TIMING_COLUMNS + COST_COLUMNS
    for col in all_columns:
        if col not in df.columns:
            df[col] = np.nan

    return df


def aggregate_single_file(df: pd.DataFrame) -> Dict[str, float]:
    """Compute per-run summary from a single file's iterations.

    Timing columns (MS_TO_S_COLUMNS) are SUMMED across iterations to give
    the total run time, matching the spreadsheet methodology.
    Non-timing metrics (cost, memory, etc.) use the LAST iteration's value
    since they represent cumulative/final state (e.g., cost_tracker uses
    running max of sum_t).
    """
    result = {}
    for col in METRIC_COLUMNS:
        if col in df.columns and df[col].notna().any():
            if col in MS_TO_S_COLUMNS:
                # Sum iterations → total run time (still in ms at this point)
                result[col] = df[col].sum()
            else:
                # Cost/memory columns: use last iteration value (cumulative)
                result[col] = df[col].dropna().iloc[-1]
        else:
            result[col] = np.nan

    # Preserve metadata using first non-null value
    if not df.empty:
        result['scaling'] = df['scaling'].dropna().iloc[0] if 'scaling' in df.columns and df['scaling'].notna().any() else ''
        result['world'] = int(df['world'].dropna().iloc[0]) if df.get('world', pd.Series(dtype=float)).notna().any() else 0
        result['rows'] = int(df['rows'].dropna().iloc[0]) if df.get('rows', pd.Series(dtype=float)).notna().any() else 0
        result['tot_l'] = int(df['tot_l'].dropna().iloc[0]) if df.get('tot_l', pd.Series(dtype=float)).notna().any() else 0

    return result


def aggregate_experiment_files(
    files: List[str],
    platform: str,
    scaling_type: str,
    instance_label: str,
    instance_detail: str,
    node_count: int,
    operation: str = "join",
    channel_type: str = "direct",
) -> Optional[Dict]:
    """Aggregate multiple run files for a single experiment configuration.

    Each file represents one run with multiple iterations.
    Returns a dict with mean/std across runs, or None if no valid data.
    """
    run_summaries = []

    for filepath in files:
        df = parse_summary_csv(filepath)
        if df.empty:
            logger.warning(f"Empty or unparseable: {filepath}")
            continue
        summary = aggregate_single_file(df)

        # Extract join_total from the corresponding scaling file
        scaling_path = find_scaling_file(filepath)
        if scaling_path:
            jt = extract_join_total(scaling_path)
            if jt is not None:
                summary['join_total'] = jt

        run_summaries.append(summary)

    if not run_summaries:
        return None

    runs_df = pd.DataFrame(run_summaries)

    result = {
        'platform': platform,
        'scaling_type': scaling_type,
        'instance_label': instance_label,
        'instance_detail': instance_detail,
        'node_count': node_count,
        'num_runs': len(run_summaries),
        'operation': operation,
        'channel_type': channel_type,
    }

    # Preserve metadata
    if 'scaling' in runs_df.columns:
        result['scaling'] = runs_df['scaling'].iloc[0]
    if 'rows' in runs_df.columns:
        result['rows'] = int(runs_df['rows'].iloc[0])
    if 'tot_l' in runs_df.columns:
        result['tot_l'] = int(runs_df['tot_l'].iloc[0])

    # Compute join_total mean/std (already in seconds, from scaling files)
    if 'join_total' in runs_df.columns and runs_df['join_total'].notna().any():
        jt_vals = runs_df['join_total'].dropna()
        result['join_total_mean'] = jt_vals.mean()
        result['join_total_std'] = jt_vals.std() if len(jt_vals) > 1 else 0.0
    else:
        result['join_total_mean'] = np.nan
        result['join_total_std'] = np.nan

    # Compute mean and std for each metric, converting ms to seconds
    for col in METRIC_COLUMNS:
        if col in runs_df.columns and runs_df[col].notna().any():
            values = runs_df[col].dropna()
            divisor = 1000.0 if col in MS_TO_S_COLUMNS else 1.0
            result[f'{col}_mean'] = values.mean() / divisor
            result[f'{col}_std'] = values.std() / divisor if len(values) > 1 else 0.0
        else:
            result[f'{col}_mean'] = np.nan
            result[f'{col}_std'] = np.nan

    # Flags for chart generator
    result['has_timing_breakdown'] = (
        pd.notna(result.get('compute_t_mean')) and
        pd.notna(result.get('comm_t_mean'))
    )
    result['has_cost_data'] = (
        (pd.notna(result.get('lambda_cost_usd_mean')) and result.get('lambda_cost_usd_mean', 0) > 0) or
        (pd.notna(result.get('step_fn_cost_usd_mean')) and result.get('step_fn_cost_usd_mean', 0) > 0) or
        (pd.notna(result.get('total_cost_usd_mean')) and result.get('total_cost_usd_mean', 0) > 0)
    )

    return result


def discover_local_files(
    local_dir: str,
    platform: str,
    scaling_type: str,
    instance_label: str,
    node_counts: List[int],
) -> Dict[int, List[str]]:
    """Discover summary files in a local directory, grouped by node count.

    Returns {node_count: [list of file paths]}.
    """
    files_by_node: Dict[int, List[str]] = {n: [] for n in node_counts}

    if not os.path.isdir(local_dir):
        logger.warning(f"Local directory not found: {local_dir}")
        return files_by_node

    # Regex for instance label: must appear after underscore, followed by _ or .
    # e.g. _16_28_ matches label "16_28", _8_ or _8. matches label "8"
    instance_re = re.compile(rf'_{re.escape(instance_label)}[_.]')
    # scaling_type short codes used in filenames (e.g. _weak_, _strong_)
    scaling_filter = f'_{scaling_type}_'
    known_scaling_tokens = ('_weak_', '_strong_', '_microbenchmark_')

    for root, _, filenames in os.walk(local_dir):
        for fname in filenames:
            if not (fname.startswith('cylon_summary_') or fname.startswith('fmi_summary_')):
                continue
            if fname.endswith('.log'):
                continue
            if not instance_re.search(fname):
                continue
            # If the filename contains a scaling type token, it must match.
            # Some older filenames omit the scaling type, so we only reject
            # on mismatch, not on absence.
            has_scaling_token = any(tok in fname for tok in known_scaling_tokens)
            if has_scaling_token and scaling_filter not in fname:
                continue

            node_count = parse_node_count_from_filename(fname)
            if node_count is None or node_count not in files_by_node:
                continue

            filepath = os.path.join(root, fname)
            files_by_node[node_count].append(filepath)

    for nc in node_counts:
        files_by_node[nc].sort()
        if files_by_node[nc]:
            logger.info(f"  {platform}/{scaling_type}/{instance_label} {nc}node: "
                        f"{len(files_by_node[nc])} files")

    return files_by_node


def aggregate_all(
    experiments: list,
    global_local_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate results for all configured experiments.

    Args:
        experiments: List of ExperimentConfig objects.
        global_local_dir: Optional global local directory override.

    Returns:
        DataFrame with one row per (platform, scaling_type, instance, node_count).
    """
    all_results = []

    for exp in experiments:
        local_dir = global_local_dir or exp.local_data_dir
        if not local_dir:
            logger.warning(f"No data source for {exp.label}, skipping")
            continue

        logger.info(f"Aggregating: {exp.label} ({exp.scaling_type} scaling)")

        files_by_node = discover_local_files(
            local_dir=local_dir,
            platform=exp.platform,
            scaling_type=exp.scaling_type,
            instance_label=exp.instance_label,
            node_counts=exp.node_counts,
        )

        for node_count in exp.node_counts:
            files = files_by_node.get(node_count, [])
            if not files:
                logger.warning(f"  No files for {node_count} nodes")
                continue

            result = aggregate_experiment_files(
                files=files,
                platform=exp.platform,
                scaling_type=exp.scaling_type,
                instance_label=exp.instance_label,
                instance_detail=exp.instance_detail,
                node_count=node_count,
                operation=getattr(exp, 'operation', 'join'),
                channel_type=getattr(exp, 'channel_type', 'direct'),
            )
            if result:
                all_results.append(result)

    if not all_results:
        logger.warning("No results aggregated")
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def save_aggregated_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save aggregated results to CSV."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved aggregated results to {output_path}")


def parse_microbenchmark_csv(filepath: str) -> pd.DataFrame:
    """Parse a microbenchmark summary file."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"Failed to parse microbenchmark {filepath}: {e}")
        return pd.DataFrame()


def aggregate_microbenchmark_files(
    files: List[str],
    node_count: int,
) -> Optional[pd.DataFrame]:
    """Aggregate microbenchmark files for a given node count.

    Returns DataFrame with mean/std of latency/bandwidth per message size,
    averaged across runs and iterations.
    """
    all_dfs = []
    for filepath in files:
        df = parse_microbenchmark_csv(filepath)
        if df.empty:
            continue
        all_dfs.append(df)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)

    # Group by message size, compute mean/std across all iterations and runs
    grouped = combined.groupby('msg_size_bytes').agg(
        barrier_latency_ms_mean=('barrier_latency_ms', 'mean'),
        barrier_latency_ms_std=('barrier_latency_ms', 'std'),
        allreduce_latency_ms_mean=('allreduce_latency_ms', 'mean'),
        allreduce_latency_ms_std=('allreduce_latency_ms', 'std'),
        allreduce_bandwidth_mbps_mean=('allreduce_bandwidth_mbps', 'mean'),
        allreduce_bandwidth_mbps_std=('allreduce_bandwidth_mbps', 'std'),
        com_init_t_mean=('com_init_t', 'mean'),
    ).reset_index()

    grouped['node_count'] = node_count
    return grouped


def aggregate_all_microbenchmarks(
    experiments: list,
    global_local_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate microbenchmark results separately (different schema)."""
    all_results = []

    for exp in experiments:
        if exp.operation != 'microbenchmark':
            continue

        local_dir = global_local_dir or exp.local_data_dir
        if not local_dir:
            continue

        logger.info(f"Aggregating microbenchmarks: {exp.label}")

        files_by_node = discover_local_files(
            local_dir=local_dir,
            platform=exp.platform,
            scaling_type=exp.scaling_type,
            instance_label=exp.instance_label,
            node_counts=exp.node_counts,
        )

        for node_count in exp.node_counts:
            files = files_by_node.get(node_count, [])
            if not files:
                continue

            result_df = aggregate_microbenchmark_files(files, node_count)
            if result_df is not None:
                all_results.append(result_df)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)