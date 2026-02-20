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
    """Weak scaling line chart with error bars — Join operation only.

    X: Parallelism (node counts)
    Y: Average execution time (s)
    One line per platform/instance.
    """
    weak = df[(df['scaling_type'] == 'weak') & (df['operation'] == 'join')]
    # Show only 'direct' channel to avoid merging Lambda's 3 channel types
    # into one line.  The infrastructure comparison chart handles cross-channel.
    if 'channel_type' in weak.columns:
        weak = weak[weak['channel_type'] == 'direct']
    if weak.empty:
        logger.info("No weak scaling join data, skipping chart")
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
        avg_times = combined.mean(axis=1).sort_index()
        node_counts = avg_times.index.values
        baseline = avg_times.loc[min(node_counts)]
        speedup = baseline / avg_times

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
    """Time composition chart: init + data_gen + execution per channel/operation.

    Shows sequential phases of a serverless run as stacked bars.
    Addresses reviewer concern C2: where does time go?

    At low node counts, execution dominates. At high node counts,
    connection init (NAT traversal) becomes a significant fraction
    for direct TCP, while Redis/S3 avoid it but pay more per collective.
    """
    # Only Lambda data has meaningful init/data_gen breakdown.
    # Exclude rows missing data_gen_t (old experiments without full breakdown).
    lambda_data = df[(df['platform'] == 'lambda') & (df['operation'] != 'microbenchmark')]
    lambda_data = lambda_data[lambda_data['data_gen_t_mean'].notna()]
    if lambda_data.empty:
        logger.info("No Lambda timing data with full breakdown, skipping time composition chart")
        return

    CHANNEL_COLORS = {
        'direct': {'init': '#d62728', 'data_gen': '#ff9896', 'exec': '#2ca02c'},
        'redis': {'init': '#1f77b4', 'data_gen': '#aec7e8', 'exec': '#ff7f0e'},
        's3': {'init': '#9467bd', 'data_gen': '#c5b0d5', 'exec': '#8c564b'},
    }

    groups = lambda_data.groupby(['operation', 'channel_type'])
    n_groups = len(groups)
    if n_groups == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.8 / n_groups
    all_nodes = sorted(lambda_data['node_count'].unique())
    x_base = np.arange(len(all_nodes))

    for idx, ((op, channel), group) in enumerate(groups):
        group = group.sort_values('node_count')
        init_times = []
        gen_times = []
        exec_times = []

        for n in all_nodes:
            row = group[group['node_count'] == n]
            if not row.empty:
                init_t = row['com_init_t_mean'].iloc[0] if pd.notna(row['com_init_t_mean'].iloc[0]) else 0
                gen_t = row['data_gen_t_mean'].iloc[0] if pd.notna(row['data_gen_t_mean'].iloc[0]) else 0
                avg_t = row['avg_t_mean'].iloc[0] if pd.notna(row['avg_t_mean'].iloc[0]) else 0
                init_times.append(init_t)
                gen_times.append(gen_t)
                exec_times.append(avg_t)
            else:
                init_times.append(0)
                gen_times.append(0)
                exec_times.append(0)

        init_times = np.array(init_times)
        gen_times = np.array(gen_times)
        exec_times = np.array(exec_times)

        offset = (idx - n_groups / 2 + 0.5) * bar_width
        x = x_base + offset

        colors = CHANNEL_COLORS.get(channel, {'init': 'gray', 'data_gen': 'lightgray', 'exec': 'darkgray'})
        label_prefix = f'{op.title()} ({channel.upper()})'

        ax.bar(x, init_times, bar_width,
               label=f'{label_prefix} - Init', color=colors['init'],
               edgecolor='black', linewidth=0.5)
        ax.bar(x, gen_times, bar_width, bottom=init_times,
               label=f'{label_prefix} - Data Gen', color=colors['data_gen'],
               edgecolor='black', linewidth=0.5)
        ax.bar(x, exec_times, bar_width, bottom=init_times + gen_times,
               label=f'{label_prefix} - Execution', color=colors['exec'],
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Serverless Execution Time Composition')
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    _save_chart(fig, config.output_dir, 'compute-vs-comm-breakdown',
                config.chart_format, config.chart_dpi)


def chart_cost_analysis(df: pd.DataFrame, config, experiments: list = None):
    """Cost analysis bar chart for serverless experiments.

    Stacked bars: lambda_cost_usd + step_fn_cost_usd per node count,
    grouped by (operation, channel_type) to avoid merging different
    workloads into a single bar.
    Addresses reviewer concern L4: cost analysis.
    """
    has_cost = df[df['has_cost_data'] == True]
    if has_cost.empty:
        logger.info("No cost data available, skipping cost analysis chart")
        return

    # Exclude microbenchmark — its cost is dominated by init, not compute
    has_cost = has_cost[has_cost['operation'] != 'microbenchmark']
    if has_cost.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    COST_COLORS = {
        ('join', 'redis'): ('red', 'salmon'),
        ('join', 's3'): ('orange', 'moccasin'),
        ('join', 'direct'): ('green', 'lightgreen'),
        ('groupby', 'direct'): ('blue', 'lightskyblue'),
    }

    groups = has_cost.groupby(['operation', 'channel_type'])
    n_groups = len(groups)
    bar_width = 0.8 / max(n_groups, 1)

    all_nodes = sorted(has_cost['node_count'].unique())
    x_base = np.arange(len(all_nodes))

    for idx, ((op, channel), group) in enumerate(groups):
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

        lambda_color, stepfn_color = COST_COLORS.get(
            (op, channel), (DEFAULT_COLORS[idx % len(DEFAULT_COLORS)], 'lightgray'))
        label_prefix = f'{op.title()} ({channel.upper()})'
        ax.bar(x, lambda_cost, bar_width, label=f'{label_prefix} - Lambda',
               color=lambda_color, edgecolor='black', linewidth=0.5)
        ax.bar(x, stepfn_cost, bar_width, bottom=lambda_cost,
               label=f'{label_prefix} - Step Fn', color=stepfn_color,
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Cost (USD)')
    ax.set_title('Serverless Execution Cost (Lambda + Step Functions)')
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    _save_chart(fig, config.output_dir, 'cost-analysis',
                config.chart_format, config.chart_dpi)





def chart_infrastructure_comparison(df: pd.DataFrame, config, experiments: list = None):
    """Infrastructure comparison: Direct vs Redis vs S3 channels.

    Line chart comparing avg execution time across channel types for Join weak scaling.
    Addresses reviewer concern L3: baseline comparisons.
    """
    # Filter to Lambda join experiments that have channel_type
    if 'channel_type' not in df.columns:
        logger.info("No channel_type column, skipping infrastructure comparison chart")
        return

    lambda_join = df[(df['platform'] == 'lambda') & (df['operation'] == 'join')]
    if lambda_join.empty:
        logger.info("No Lambda join data, skipping infrastructure comparison chart")
        return

    channels = lambda_join['channel_type'].unique()
    if len(channels) < 2:
        logger.info("Need at least 2 channel types for comparison, skipping")
        return

    CHANNEL_STYLES = {
        'direct': ('green', 's', 'Direct (TCP)'),
        'redis': ('red', '^', 'Redis'),
        's3': ('orange', 'D', 'S3'),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for channel in sorted(channels):
        ch_data = lambda_join[lambda_join['channel_type'] == channel].sort_values('node_count')
        if ch_data.empty:
            continue

        color, marker, label = CHANNEL_STYLES.get(channel, ('gray', 'x', channel))
        ax.plot(ch_data['node_count'].astype(str), ch_data['avg_t_mean'],
                marker=marker, color=color, label=label, linewidth=2)
        ax.errorbar(ch_data['node_count'].astype(str), ch_data['avg_t_mean'],
                     yerr=ch_data['avg_t_std'], fmt='none', color=color,
                     ecolor=color, capsize=5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Communication Infrastructure Comparison (Join Weak Scaling)')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    _save_chart(fig, config.output_dir, 'infrastructure-comparison',
                config.chart_format, config.chart_dpi)


def chart_groupby_weak_scaling(df: pd.DataFrame, config, experiments: list = None):
    """GroupBy weak scaling line chart.

    Addresses reviewer concern L2: evaluation scope beyond Join.
    """
    groupby = df[(df['operation'] == 'groupby') & (df['scaling_type'] == 'weak')]
    if groupby.empty:
        logger.info("No GroupBy weak scaling data, skipping chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = groupby.groupby(['platform', 'instance_label', 'instance_detail'])
    for idx, ((platform, instance, detail), group) in enumerate(groups):
        group = group.sort_values('node_count')
        color, marker = _get_series_style(idx, experiments, platform, instance)

        label = _series_label(platform, detail, instance)
        ax.plot(group['node_count'].astype(str), group['avg_t_mean'],
                marker=marker, color=color, label=label, linewidth=2)
        ax.errorbar(group['node_count'].astype(str), group['avg_t_mean'],
                     yerr=group['avg_t_std'], fmt='none', color=color,
                     ecolor=color, capsize=5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Weak Scaling of GroupBy Operation')
    ax.legend()
    fig.tight_layout()
    _save_chart(fig, config.output_dir, 'groupby-w-scaling',
                config.chart_format, config.chart_dpi)


def chart_microbenchmark(micro_df: pd.DataFrame, config):
    """Microbenchmark charts: AllReduce latency vs message size per node count.

    Addresses reviewer concerns L2 (evaluation scope) and C2 (communication overhead).
    """
    if micro_df.empty:
        logger.info("No microbenchmark data, skipping chart")
        return

    NODE_COLORS = {
        1: 'blue', 2: 'green', 4: 'red', 8: 'orange',
        16: 'purple', 32: 'brown', 64: 'black',
    }

    # Chart 1: AllReduce latency vs message size
    fig, ax = plt.subplots(figsize=(10, 6))

    for node_count in sorted(micro_df['node_count'].unique()):
        nd = micro_df[micro_df['node_count'] == node_count].sort_values('msg_size_bytes')
        color = NODE_COLORS.get(node_count, 'gray')
        ax.plot(nd['msg_size_bytes'], nd['allreduce_latency_ms_mean'],
                marker='o', color=color, label=f'{node_count} nodes', linewidth=2)
        if 'allreduce_latency_ms_std' in nd.columns:
            ax.fill_between(nd['msg_size_bytes'],
                           nd['allreduce_latency_ms_mean'] - nd['allreduce_latency_ms_std'],
                           nd['allreduce_latency_ms_mean'] + nd['allreduce_latency_ms_std'],
                           alpha=0.2, color=color)

    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('AllReduce Latency (ms)')
    ax.set_title('AllReduce Latency vs Message Size (Lambda Direct)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    _save_chart(fig, config.output_dir, 'microbenchmark-allreduce-latency',
                config.chart_format, config.chart_dpi)

    # Chart 2: Barrier latency vs node count
    barrier = micro_df.groupby('node_count').agg(
        barrier_mean=('barrier_latency_ms_mean', 'mean'),
        barrier_std=('barrier_latency_ms_std', 'mean'),
    ).reset_index().sort_values('node_count')

    if not barrier.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(barrier['node_count'].astype(str), barrier['barrier_mean'],
               yerr=barrier['barrier_std'], color='steelblue',
               edgecolor='black', linewidth=0.5, capsize=5)
        ax.set_xlabel('Parallelism (Nodes)')
        ax.set_ylabel('Barrier Latency (ms)')
        ax.set_title('Barrier Latency vs Node Count (Lambda Direct)')
        fig.tight_layout()
        _save_chart(fig, config.output_dir, 'microbenchmark-barrier-latency',
                    config.chart_format, config.chart_dpi)


def chart_cost_per_operation(df: pd.DataFrame, config, experiments: list = None):
    """Cost comparison across operations and channel types.

    Shows total_cost_usd for each (operation, channel_type) at each node count.
    """
    has_cost = df[df['has_cost_data'] == True]
    if has_cost.empty or 'channel_type' not in has_cost.columns:
        return

    lambda_cost = has_cost[has_cost['platform'] == 'lambda']
    if lambda_cost.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = lambda_cost.groupby(['operation', 'channel_type'])
    n_groups = len(groups)
    bar_width = 0.8 / max(n_groups, 1)
    all_nodes = sorted(lambda_cost['node_count'].unique())
    x_base = np.arange(len(all_nodes))

    OP_COLORS = {
        ('join', 'direct'): 'green',
        ('join', 'redis'): 'red',
        ('join', 's3'): 'orange',
        ('groupby', 'direct'): 'blue',
    }

    for idx, ((op, channel), group) in enumerate(groups):
        group = group.sort_values('node_count')
        costs = []
        for n in all_nodes:
            row = group[group['node_count'] == n]
            costs.append(row['total_cost_usd_mean'].iloc[0] if not row.empty else 0)

        offset = (idx - n_groups / 2 + 0.5) * bar_width
        x = x_base + offset
        color = OP_COLORS.get((op, channel), DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
        ax.bar(x, costs, bar_width, label=f'{op.title()} ({channel.upper()})',
               color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Cost (USD)')
    ax.set_title('Lambda Execution Cost by Operation and Channel Type')
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(n) for n in all_nodes])
    ax.legend()
    fig.tight_layout()
    _save_chart(fig, config.output_dir, 'cost-per-operation',
                config.chart_format, config.chart_dpi)


def generate_all_charts(df: pd.DataFrame, config, micro_df: pd.DataFrame = None) -> None:
    """Generate all charts from aggregated data."""
    os.makedirs(config.output_dir, exist_ok=True)

    experiments = config.experiments if hasattr(config, 'experiments') else None

    # Existing charts
    chart_weak_scaling(df, config, experiments)
    chart_strong_scaling(df, config, experiments)
    chart_strong_scaling_with_speedup(df, config, experiments)
    chart_strong_scaling_scaled(df, config, experiments)

    # Reviewer concern charts
    chart_compute_vs_comm_breakdown(df, config, experiments)
    chart_cost_analysis(df, config, experiments)
    chart_infrastructure_comparison(df, config, experiments)
    chart_groupby_weak_scaling(df, config, experiments)
    chart_cost_per_operation(df, config, experiments)

    # Microbenchmark charts (separate schema)
    if micro_df is not None and not micro_df.empty:
        chart_microbenchmark(micro_df, config)

    logger.info(f"Generated charts in {config.output_dir}")