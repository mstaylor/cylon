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
Notebook generator for Cylon experiment results.

Generates a Jupyter notebook (frontiersCloudSubmission.ipynb) with cells
for loading aggregated data and producing each chart type. The user can
then run cells interactively and tweak charts before exporting for the paper.
"""

import logging
import os

import nbformat

logger = logging.getLogger(__name__)

NBFORMAT_VERSION = 4

# Platform display names (must match chart_generator.py)
PLATFORM_NAMES_CODE = """\
PLATFORM_NAMES = {
    'ec2': 'EC2',
    'ecs': 'ECS',
    'fargate': 'Fargate',
    'rivanna': 'Rivanna',
    'lambda': 'Lambda',
}
DEFAULT_COLORS = ['blue', 'green', 'red', 'orange', 'black', 'purple', 'cyan', 'brown']
DEFAULT_MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']


def _platform_name(platform):
    return PLATFORM_NAMES.get(platform, platform.capitalize())


def _series_label(platform, instance_detail=None, instance_label=None):
    name = _platform_name(platform)
    detail = instance_detail if instance_detail else instance_label
    return f"{name} - {detail}"


def _get_series_style(idx, platform=None, instance=None, config_map=None):
    if config_map and (platform, instance) in config_map:
        ec = config_map[(platform, instance)]
        return ec.get('color', DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]), \\
               ec.get('marker', DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)])
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)], DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)]
"""


def _make_markdown_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=source)


def _make_code_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=source)


def generate_notebook(aggregated_csv_path: str, output_path: str,
                      output_chart_dir: str = None) -> str:
    """Generate a Jupyter notebook with all chart cells.

    Args:
        aggregated_csv_path: Path to the aggregated_results.csv.
        output_path: Where to write the .ipynb file.
        output_chart_dir: Directory for saving chart files from notebook cells.
                         If None, defaults to the directory containing the notebook.

    Returns:
        The output_path written.
    """
    nb = nbformat.v4.new_notebook()
    nb.metadata.update({
        'kernelspec': {
            'display_name': 'Python 3 (ipykernel)',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0',
        },
    })

    if output_chart_dir is None:
        output_chart_dir = os.path.dirname(output_path) or '.'

    cells = []

    # --- Title ---
    cells.append(_make_markdown_cell(
        "# Frontiers Cloud Submission - Experiment Results\n\n"
        "This notebook generates publication-quality charts from aggregated Cylon experiment data.\n\n"
        "**Pipeline**: `target/shared/scripts/results/pipeline.py`  \n"
        "**Data**: `aggregated_results.csv` (produced by the aggregate step)"
    ))

    # --- Imports ---
    cells.append(_make_code_cell(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import os\n"
        "\n"
        "%matplotlib inline\n"
        "plt.rcParams['figure.dpi'] = 150"
    ))

    # --- Helper functions ---
    cells.append(_make_markdown_cell("## Helper Functions"))
    cells.append(_make_code_cell(PLATFORM_NAMES_CODE))

    # --- Load data ---
    cells.append(_make_markdown_cell("## Load Aggregated Data"))
    cells.append(_make_code_cell(
        f"CSV_PATH = r'{aggregated_csv_path}'\n"
        f"CHART_DIR = r'{output_chart_dir}'\n"
        "os.makedirs(CHART_DIR, exist_ok=True)\n"
        "\n"
        "df = pd.read_csv(CSV_PATH)\n"
        "print(f'Loaded {len(df)} rows')\n"
        "df.head()"
    ))

    # --- Chart save helper ---
    cells.append(_make_code_cell(
        "def save_chart(fig, name, fmt='svg', dpi=300):\n"
        "    path = os.path.join(CHART_DIR, f'{name}.{fmt}')\n"
        "    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')\n"
        "    print(f'Saved: {path}')"
    ))

    # --- Weak Scaling ---
    cells.append(_make_markdown_cell("## Weak Scaling of Join Operation"))
    cells.append(_make_code_cell(_cell_weak_scaling()))

    # --- Strong Scaling ---
    cells.append(_make_markdown_cell("## Strong Scaling of Join Operation"))
    cells.append(_make_code_cell(_cell_strong_scaling()))

    # --- Strong Scaling with Speedup ---
    cells.append(_make_markdown_cell("## Strong Scaling with Speedup (Dual Axis)"))
    cells.append(_make_code_cell(_cell_strong_scaling_speedup()))

    # --- Strong Scaling Scaled ---
    cells.append(_make_markdown_cell("## Strong Scaling Scaled by Time (time * nodes)"))
    cells.append(_make_code_cell(_cell_strong_scaling_scaled()))

    # --- Compute vs Communication Breakdown ---
    cells.append(_make_markdown_cell(
        "## Compute vs Communication Time Breakdown\n\n"
        "*Addresses reviewer concern C2*"
    ))
    cells.append(_make_code_cell(_cell_compute_vs_comm()))

    # --- Cost Analysis ---
    cells.append(_make_markdown_cell(
        "## Serverless Execution Cost Analysis\n\n"
        "*Addresses reviewer concern L4*"
    ))
    cells.append(_make_code_cell(_cell_cost_analysis()))

    nb.cells = cells

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        nbformat.write(nb, f)

    logger.info(f"Generated notebook: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Cell source generators
# ---------------------------------------------------------------------------

def _cell_weak_scaling() -> str:
    return """\
weak = df[df['scaling_type'] == 'weak']
if weak.empty:
    print("No weak scaling data")
else:
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = weak.groupby(['platform', 'instance_label', 'instance_detail'])
    for idx, ((platform, instance, detail), group) in enumerate(groups):
        group = group.sort_values('node_count')
        color, marker = _get_series_style(idx)

        label = _series_label(platform, detail, instance)
        ax.plot(group['node_count'].astype(str), group['avg_t_mean'],
                marker=marker, color=color, label=label)
        ax.errorbar(group['node_count'].astype(str), group['avg_t_mean'],
                     yerr=group['avg_t_std'], fmt='x', color=color,
                     ecolor=color, capsize=5)

    ax.set_xlabel('Parallelism (Nodes)')
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Weak Scaling of Join Operation')
    n_legend_rows = (len(groups) + 1) // 2
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08 + n_legend_rows * 0.05)
    save_chart(fig, 'join-w-scaling')
    plt.show()
"""


def _cell_strong_scaling() -> str:
    return """\
strong = df[df['scaling_type'] == 'strong']
if strong.empty:
    print("No strong scaling data")
else:
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = strong.groupby(['platform', 'instance_label', 'instance_detail'])
    for idx, ((platform, instance, detail), group) in enumerate(groups):
        group = group.sort_values('node_count')
        color, marker = _get_series_style(idx)

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
    save_chart(fig, 'join-s-scaling')
    plt.show()
"""


def _cell_strong_scaling_speedup() -> str:
    return """\
strong = df[df['scaling_type'] == 'strong']
if strong.empty:
    print("No strong scaling data")
else:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    groups = strong.groupby(['platform', 'instance_label', 'instance_detail'])
    all_series = []

    for idx, ((platform, instance, detail), group) in enumerate(groups):
        group = group.sort_values('node_count')
        color, marker = _get_series_style(idx)

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
    save_chart(fig, 'join-s-scaling-speedup')
    plt.show()
"""


def _cell_strong_scaling_scaled() -> str:
    return """\
strong = df[df['scaling_type'] == 'strong']
if strong.empty:
    print("No strong scaling data")
else:
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = strong.groupby(['platform', 'instance_label', 'instance_detail'])
    for idx, ((platform, instance, detail), group) in enumerate(groups):
        group = group.sort_values('node_count')
        color, marker = _get_series_style(idx)

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
    save_chart(fig, 'join-s-scaling-scaled')
    plt.show()
"""



def _cell_compute_vs_comm() -> str:
    return """\
has_breakdown = df[df['has_timing_breakdown'] == True]
if has_breakdown.empty:
    print("No timing breakdown data available (needs data_gen_t, compute_t, comm_t columns)")
else:
    groups = has_breakdown.groupby(['platform', 'instance_label', 'instance_detail'])
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(12, 7))

    bar_width = 0.8 / n_groups
    group_list = list(groups)

    all_nodes = sorted(has_breakdown['node_count'].unique())
    x_base = np.arange(len(all_nodes))

    for idx, ((platform, instance, detail), group) in enumerate(group_list):
        group = group.sort_values('node_count')
        data_gen = []
        compute = []
        comm = []
        for n in all_nodes:
            row = group[group['node_count'] == n]
            if not row.empty:
                data_gen.append(row['data_gen_t_mean'].iloc[0])
                compute.append(row['compute_t_mean'].iloc[0])
                comm.append(row['comm_t_mean'].iloc[0])
            else:
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
    save_chart(fig, 'compute-vs-comm-breakdown')
    plt.show()
"""


def _cell_cost_analysis() -> str:
    return """\
has_cost = df[df['has_cost_data'] == True]
if has_cost.empty:
    print("No cost data available (needs lambda_cost_usd columns)")
else:
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
    save_chart(fig, 'cost-analysis')
    plt.show()
"""