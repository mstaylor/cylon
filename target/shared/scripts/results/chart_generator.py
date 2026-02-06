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
Chart generator for Cylon experiment results.

Generates publication-quality scaling charts from aggregated CSV data.
Replicates existing notebook chart styles and adds new charts for
reviewer concerns (compute/comm breakdown, cost analysis, microbenchmarks).
"""

import logging
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default colors if not specified in config
DEFAULT_COLORS = ['blue', 'green', 'red', 'orange', 'black', 'purple', 'cyan', 'brown']
DEFAULT_MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

# Platform display names (acronyms stay uppercase, proper nouns title-cased)
PLATFORM_NAMES = {
    'ec2': 'EC2',
    'ecs': 'ECS',
    'fargate': 'Fargate',
    'rivanna': 'Rivanna',
    'lambda': 'Lambda',
}


def _platform_name(platform: str) -> str:
    """Get display name for a platform."""
    return PLATFORM_NAMES.get(platform, platform.capitalize())


def _series_label(platform: str, instance_detail: str = None, instance_label: str = None) -> str:
    """Build a chart legend label like 'EC2 - 16 CPU/28 GB Memory'."""
    name = _platform_name(platform)
    detail = instance_detail if instance_detail else instance_label
    return f"{name} - {detail}"


def _get_series_style(idx: int, exp_configs: list = None, platform: str = None,
                      instance: str = None):
    """Get color and marker for a series."""
    if exp_configs:
        for ec in exp_configs:
            if ec.platform == platform and ec.instance_label == instance:
                return ec.color, ec.marker
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)], DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)]


def _save_chart(fig, output_dir: str, name: str, fmt: str = 'svg', dpi: int = 300):
    """Save chart to file."""
    path = os.path.join(output_dir, f'{name}.{fmt}')
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved chart: {path}")


def chart_weak_scaling(df: pd.DataFrame, config, experiments: list = None):
    """Weak scaling line chart with error bars.

    X: Parallelism (node counts)
    Y: Average execution time (s)
    One line per platform/instance.
    """
    weak = df[df['scaling_type'] == 'weak']
    if weak.empty:
        logger.info("No weak scaling data, skipping chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = weak.groupby(['platform', 'instance_label', 'instance_detail'])
    for idx, ((platform, instance, detail), group) in enumerate(groups):
        group = group.sort_values('node_count')
        color, marker = _get_series_style(idx, experiments, platform, instance)

        label = _series_label(platform, detail, instance)
        ax.plot(group['node_count'].astype(str), group['avg_t_mean'],
                marker=marker, color=color, label=label)
        ax.errorbar(group['node_count'].astype(str), group['avg_t_mean'],
                     yerr=group['avg_t_std'], fmt='x', color=color,
                     ecolor=color, capsize=5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Weak Scaling of Join Operation')
    n_legend_rows = (len(groups) + 1) // 2  # 2 columns
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08 + n_legend_rows * 0.05)
    _save_chart(fig, config.output_dir, 'join-w-scaling', config.chart_format, config.chart_dpi)


def chart_strong_scaling(df: pd.DataFrame, config, experiments: list = None):
    """Strong scaling line chart with error bars.

    X: Parallelism (node counts)
    Y: Average execution time (s)
    One line per platform/instance.
    """
    strong = df[df['scaling_type'] == 'strong']
    if strong.empty:
        logger.info("No strong scaling data, skipping chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = strong.groupby(['platform', 'instance_label', 'instance_detail'])
    for idx, ((platform, instance, detail), group) in enumerate(groups):
        group = group.sort_values('node_count')
        color, marker = _get_series_style(idx, experiments, platform, instance)

        label = _series_label(platform, detail, instance)
        ax.plot(group['node_count'].astype(str), group['avg_t_mean'],
                marker=marker, color=color, label=label)
        ax.errorbar(group['node_count'].astype(str), group['avg_t_mean'],
                     yerr=group['avg_t_std'], fmt='x', color=color,
                     ecolor=color, capsize=5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Strong Scaling of Join Operation')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save_chart(fig, config.output_dir, 'join-s-scaling', config.chart_format, config.chart_dpi)


def chart_strong_scaling_with_speedup(df: pd.DataFrame, config, experiments: list = None):
    """Strong scaling with dual-axis: execution time + speedup.

    Left Y: Average execution time (s)
    Right Y: Speedup (T_1 / T_p averaged across platforms)
    """
    strong = df[df['scaling_type'] == 'strong']
    if strong.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    groups = strong.groupby(['platform', 'instance_label', 'instance_detail'])
    all_series = []

    for idx, ((platform, instance, detail), group) in enumerate(groups):
        group = group.sort_values('node_count')
        color, marker = _get_series_style(idx, experiments, platform, instance)

        label = _series_label(platform, detail, instance)
        ax1.plot(group['node_count'].astype(str), group['avg_t_mean'],
                 marker=marker, color=color, label=label)
        ax1.errorbar(group['node_count'].astype(str), group['avg_t_mean'],
                      yerr=group['avg_t_std'], fmt='x', color=color,
                      ecolor=color, capsize=5)
        all_series.append(group[['node_count', 'avg_t_mean']].set_index('node_count'))

    ax1.set_xlabel('Parallelism (Nodes)')
    ax1.set_ylabel('Average Execution Time (s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Strong Scaling of Join Operation')

    # Compute average speedup across all platforms
    if all_series:
        combined = pd.concat(all_series, axis=1)
        avg_times = combined.mean(axis=1)
        node_counts = avg_times.index.values
        speedup = avg_times.iloc[0] / avg_times
        speedup.iloc[0] = 1.0

        ax2 = ax1.twinx()
        ax2.plot([str(n) for n in node_counts], speedup.values, 'o--',
                 color='blue', label='Speedup (Avg)')
        ax2.set_ylabel('Speedup', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save_chart(fig, config.output_dir, 'join-s-scaling-speedup',
                config.chart_format, config.chart_dpi)


def chart_strong_scaling_scaled(df: pd.DataFrame, config, experiments: list = None):
    """Strong scaling chart with time scaled by node count (time * nodes).

    Shows parallel overhead - ideal scaling would be a flat line.
    """
    strong = df[df['scaling_type'] == 'strong']
    if strong.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = strong.groupby(['platform', 'instance_label', 'instance_detail'])
    for idx, ((platform, instance, detail), group) in enumerate(groups):
        group = group.sort_values('node_count')
        color, marker = _get_series_style(idx, experiments, platform, instance)

        scaled_time = group['avg_t_mean'] * group['node_count']
        label = _series_label(platform, detail, instance)
        ax.plot(group['node_count'].astype(str), scaled_time,
                marker=marker, color=color, label=label)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Strong Scaling of Join Operation Scaled by Time(s)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save_chart(fig, config.output_dir, 'join-s-scaling-scaled',
                config.chart_format, config.chart_dpi)


def chart_compute_vs_comm_breakdown(df: pd.DataFrame, config, experiments: list = None):
    """Stacked bar chart: data_gen_t / compute_t / comm_t per platform and node count.

    Addresses reviewer concern C2: compute vs communication breakdown.
    """
    has_breakdown = df[df['has_timing_breakdown'] == True]
    if has_breakdown.empty:
        logger.info("No timing breakdown data available, skipping compute/comm chart")
        return

    groups = has_breakdown.groupby(['platform', 'instance_label', 'instance_detail'])
    n_groups = len(groups)
    if n_groups == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    bar_width = 0.8 / n_groups
    group_list = list(groups)

    # Get all unique node counts across groups
    all_nodes = sorted(has_breakdown['node_count'].unique())
    x_base = np.arange(len(all_nodes))

    for idx, ((platform, instance, detail), group) in enumerate(group_list):
        group = group.sort_values('node_count')
        # Align to all_nodes positions
        x_positions = []
        data_gen = []
        compute = []
        comm = []
        for n in all_nodes:
            row = group[group['node_count'] == n]
            if not row.empty:
                x_positions.append(True)
                data_gen.append(row['data_gen_t_mean'].iloc[0])
                compute.append(row['compute_t_mean'].iloc[0])
                comm.append(row['comm_t_mean'].iloc[0])
            else:
                x_positions.append(False)
                data_gen.append(0)
                compute.append(0)
                comm.append(0)

        data_gen = np.array(data_gen)
        compute = np.array(compute)
        comm = np.array(comm)

        offset = (idx - n_groups / 2 + 0.5) * bar_width
        x = x_base + offset

        label_prefix = _series_label(platform, detail, instance)
        ax.bar(x, data_gen, bar_width, label=f'{label_prefix} - Data Gen',
               color='lightblue', edgecolor='black', linewidth=0.5)
        ax.bar(x, compute, bar_width, bottom=data_gen,
               label=f'{label_prefix} - Compute', color='green',
               edgecolor='black', linewidth=0.5)
        ax.bar(x, comm, bar_width, bottom=data_gen + compute,
               label=f'{label_prefix} - Communication', color='orange',
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Compute vs Communication Time Breakdown')
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    _save_chart(fig, config.output_dir, 'compute-vs-comm-breakdown',
                config.chart_format, config.chart_dpi)


def chart_cost_analysis(df: pd.DataFrame, config, experiments: list = None):
    """Cost analysis bar chart for serverless experiments.

    Stacked bars: lambda_cost_usd + step_fn_cost_usd per node count.
    Addresses reviewer concern L4: cost analysis.
    """
    has_cost = df[df['has_cost_data'] == True]
    if has_cost.empty:
        logger.info("No cost data available, skipping cost analysis chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = has_cost.groupby(['platform', 'instance_label', 'instance_detail'])
    n_groups = len(groups)
    bar_width = 0.8 / max(n_groups, 1)
    group_list = list(groups)

    all_nodes = sorted(has_cost['node_count'].unique())
    x_base = np.arange(len(all_nodes))

    for idx, ((platform, instance, detail), group) in enumerate(group_list):
        group = group.sort_values('node_count')
        lambda_cost = []
        stepfn_cost = []
        for n in all_nodes:
            row = group[group['node_count'] == n]
            if not row.empty:
                lambda_cost.append(row['lambda_cost_usd_mean'].iloc[0])
                stepfn_cost.append(row['step_fn_cost_usd_mean'].iloc[0])
            else:
                lambda_cost.append(0)
                stepfn_cost.append(0)

        lambda_cost = np.array(lambda_cost)
        stepfn_cost = np.array(stepfn_cost)

        offset = (idx - n_groups / 2 + 0.5) * bar_width
        x = x_base + offset

        label_prefix = _series_label(platform, detail, instance)
        ax.bar(x, lambda_cost, bar_width, label=f'{label_prefix} - Lambda',
               color='blue', edgecolor='black', linewidth=0.5)
        ax.bar(x, stepfn_cost, bar_width, bottom=lambda_cost,
               label=f'{label_prefix} - Step Functions', color='orange',
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Cost (USD)')
    ax.set_title('Serverless Execution Cost Analysis')
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save_chart(fig, config.output_dir, 'cost-analysis',
                config.chart_format, config.chart_dpi)


def chart_infrastructure_comparison(df: pd.DataFrame, config, experiments: list = None):
    """Bar chart comparing single-node performance across platforms."""
    single_node = df[df['node_count'] == 1]
    if single_node.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    labels = []
    times = []
    errors = []
    for _, row in single_node.iterrows():
        detail = row.get('instance_detail', row['instance_label'])
        labels.append(f"{_platform_name(row['platform'])} {detail}")
        times.append(row['avg_t_mean'])
        errors.append(row['avg_t_std'])

    p1 = ax.bar(labels, errors, color='orange', label='Error (Std Dev)')
    p2 = ax.bar(labels, times, bottom=errors, color='blue', label='Join Time')

    ax.set_xlabel('Infrastructure')
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Single Node Join Performance Comparison')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    _save_chart(fig, config.output_dir, 'infrastructure-comparison',
                config.chart_format, config.chart_dpi)


def generate_all_charts(df: pd.DataFrame, config) -> None:
    """Generate all charts from aggregated data."""
    os.makedirs(config.output_dir, exist_ok=True)

    experiments = config.experiments if hasattr(config, 'experiments') else None

    chart_weak_scaling(df, config, experiments)
    chart_strong_scaling(df, config, experiments)
    chart_strong_scaling_with_speedup(df, config, experiments)
    chart_strong_scaling_scaled(df, config, experiments)
    chart_compute_vs_comm_breakdown(df, config, experiments)
    chart_cost_analysis(df, config, experiments)
    chart_infrastructure_comparison(df, config, experiments)

    logger.info(f"Generated charts in {config.output_dir}")