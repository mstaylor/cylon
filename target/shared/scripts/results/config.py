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
Configuration for the Cylon experiment results pipeline.

Supports YAML config files and CLI arguments.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# CSV column definitions
LEGACY_COLUMNS = [
    "scaling", "world", "rows", "max_value", "rank",
    "avg_t", "tot_l", "elapsed_t", "max_t", "com_init_t", "barrier_t"
]

TIMING_COLUMNS = ["data_gen_t", "compute_t", "comm_t"]

COST_COLUMNS = [
    "lambda_memory_mb", "lambda_duration_ms", "lambda_gb_seconds",
    "lambda_cost_usd", "step_fn_cost_usd", "total_cost_usd"
]

MICROBENCHMARK_COLUMNS = [
    "world", "iteration", "com_init_t",
    "barrier_latency_ms", "msg_size_bytes",
    "allreduce_latency_ms", "allreduce_bandwidth_mbps"
]

# Metric columns to aggregate (compute mean/std)
METRIC_COLUMNS = [
    "avg_t", "elapsed_t", "max_t", "com_init_t", "barrier_t",
    "data_gen_t", "compute_t", "comm_t",
    "lambda_memory_mb", "lambda_duration_ms", "lambda_gb_seconds",
    "lambda_cost_usd", "step_fn_cost_usd", "total_cost_usd"
]


@dataclass
class ExperimentConfig:
    """Defines one experiment: platform + scaling type + instance config."""
    platform: str               # "ec2", "ecs", "fargate", "rivanna", "lambda"
    scaling_type: str           # "weak", "strong", "microbenchmark"
    instance_label: str         # "m3xlarge", "m3large", "10GB", etc.
    instance_detail: str        # Human-readable: "15GB Mem 4 vCPUs"
    node_counts: List[int]      # [1, 2, 4, 8, 16] or [1, 2, 4, 8, 16, 32, 64]
    rows: int                   # e.g., 9100000 for weak, 145000000 for strong
    operation: str = "join"     # "join", "groupby", "microbenchmark"
    channel_type: str = "direct"  # "direct", "redis", "s3"
    color: str = "blue"
    marker: str = "o"

    # Data source - one of these must be set
    s3_prefix_pattern: Optional[str] = None  # S3 prefix template
    local_data_dir: Optional[str] = None     # Local directory with summary CSVs

    @property
    def label(self) -> str:
        return f"{self.platform.upper()} - {self.instance_detail}"

    @property
    def sheet_name(self) -> str:
        return f"{self.platform.upper()} {self.scaling_type.capitalize()}"


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    s3_bucket: str = ""
    download_dir: str = "./data/raw"
    output_dir: str = "./output"
    chart_format: str = "svg"
    chart_dpi: int = 300
    experiments: List[ExperimentConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        experiments = []
        for exp_data in data.get('experiments', []):
            experiments.append(ExperimentConfig(**exp_data))

        return cls(
            s3_bucket=data.get('s3_bucket', ''),
            download_dir=data.get('download_dir', './data/raw'),
            output_dir=data.get('output_dir', './output'),
            chart_format=data.get('chart_format', 'svg'),
            chart_dpi=data.get('chart_dpi', 300),
            experiments=experiments,
        )

    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        """Build config from CLI arguments for single-experiment mode."""
        exp = ExperimentConfig(
            platform=args.platform,
            scaling_type=args.scaling,
            instance_label=args.instance,
            instance_detail=args.instance,
            node_counts=[int(n) for n in args.nodes.split(',')],
            rows=args.rows,
            operation=args.operation,
            s3_prefix_pattern=getattr(args, 's3_prefix', None),
            local_data_dir=getattr(args, 'local_dir', None),
        )
        return cls(
            s3_bucket=getattr(args, 'bucket', '') or '',
            download_dir=getattr(args, 'download_dir', './data/raw'),
            output_dir=getattr(args, 'output_dir', './output'),
            experiments=[exp],
        )