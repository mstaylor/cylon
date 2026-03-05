#!/usr/bin/env python3
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
Cylon Experiment Results Pipeline

Orchestrates: download -> aggregate -> charts

Usage:
    # Full pipeline from YAML config
    python -m results.pipeline --config configs/experiment_config.yaml

    # From local data only (skip S3)
    python -m results.pipeline --config configs/experiment_config.yaml --local-dir /path/to/results/

    # Run individual steps
    python -m results.pipeline --config configs/experiment_config.yaml --step download
    python -m results.pipeline --config configs/experiment_config.yaml --step aggregate
    python -m results.pipeline --config configs/experiment_config.yaml --step charts

    # Quick single-experiment mode
    python -m results.pipeline --platform ec2 --scaling strong --instance 16_28 \\
        --rows 145000000 --nodes 1,2,4,8,16 --local-dir /path/to/results/
"""

import argparse
import logging
import os
import sys

from .config import PipelineConfig
from .results_downloader import download_experiment_results
from .results_aggregator import aggregate_all, aggregate_all_microbenchmarks, save_aggregated_csv
from .chart_generator import generate_all_charts
from .notebook_generator import generate_notebook

logger = logging.getLogger(__name__)

STEPS = ['download', 'aggregate', 'charts', 'notebook']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Cylon experiment results pipeline',
    )

    # Config file mode
    parser.add_argument('--config', type=str, help='YAML config file path')

    # Single-experiment CLI mode
    parser.add_argument('--platform', type=str, help='Platform name (ec2, fargate, rivanna, lambda)')
    parser.add_argument('--scaling', type=str, help='Scaling type (weak, strong)')
    parser.add_argument('--instance', type=str, help='Instance label (16_28, 4_26, 8, etc.)')
    parser.add_argument('--rows', type=int, default=0, help='Row count')
    parser.add_argument('--nodes', type=str, help='Comma-separated node counts (1,2,4,8,16)')
    parser.add_argument('--operation', type=str, default='join', help='Operation (join, groupby)')

    # Data source
    parser.add_argument('--bucket', type=str, help='S3 bucket name')
    parser.add_argument('--s3-prefix', type=str, help='S3 prefix pattern')
    parser.add_argument('--local-dir', type=str, help='Local directory with summary files')

    # Output
    parser.add_argument('--download-dir', type=str, default='./data/raw', help='Download directory')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--chart-format', type=str, default='svg', choices=['svg', 'png'], help='Chart format')
    parser.add_argument('--chart-dpi', type=int, default=None, help='Chart DPI (default: from config or 300)')

    # Notebook
    parser.add_argument('--notebook-name', type=str, default='frontiersCloudSubmission',
                        help='Name of the generated notebook (without .ipynb)')

    # Steps
    parser.add_argument('--step', type=str, action='append', choices=STEPS,
                        help='Run specific step(s). Default: all steps.')

    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    return parser


def run_pipeline(config: PipelineConfig, steps: list, local_dir: str = None) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    aggregated_csv = os.path.join(config.output_dir, 'aggregated_results.csv')

    # Step 1: Download
    if 'download' in steps:
        logger.info("=== Step: Download ===")
        download_experiment_results(config)

    # Step 2: Aggregate
    micro_csv = os.path.join(config.output_dir, 'microbenchmark_results.csv')
    if 'aggregate' in steps:
        logger.info("=== Step: Aggregate ===")
        df = aggregate_all(config.experiments, global_local_dir=local_dir)
        if df.empty:
            logger.error("No data aggregated. Check your config and data paths.")
            return
        save_aggregated_csv(df, aggregated_csv)
        logger.info(f"Aggregated {len(df)} experiment configurations")

        # Aggregate microbenchmarks separately (different schema)
        micro_df = aggregate_all_microbenchmarks(config.experiments, global_local_dir=local_dir)
        if not micro_df.empty:
            save_aggregated_csv(micro_df, micro_csv)
            logger.info(f"Aggregated {len(micro_df)} microbenchmark entries")

    # Step 3: Charts (generate image files directly)
    if 'charts' in steps:
        logger.info("=== Step: Charts ===")
        import pandas as pd
        if not os.path.exists(aggregated_csv):
            logger.error(f"Aggregated CSV not found: {aggregated_csv}. Run 'aggregate' step first.")
            return
        df = pd.read_csv(aggregated_csv)
        micro_df = pd.read_csv(micro_csv) if os.path.exists(micro_csv) else None
        generate_all_charts(df, config, micro_df=micro_df)
        logger.info(f"Charts saved to {config.output_dir}")

    # Step 4: Notebook (generate Jupyter notebook with chart cells)
    if 'notebook' in steps:
        logger.info("=== Step: Notebook ===")
        if not os.path.exists(aggregated_csv):
            logger.error(f"Aggregated CSV not found: {aggregated_csv}. Run 'aggregate' step first.")
            return
        notebook_name = getattr(config, 'notebook_name', 'frontiersCloudSubmission')
        notebook_path = os.path.join(config.output_dir, f'{notebook_name}.ipynb')
        generate_notebook(
            aggregated_csv_path=aggregated_csv,
            output_path=notebook_path,
            output_chart_dir=config.output_dir,
        )
        logger.info(f"Notebook saved to {notebook_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    # Build config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    elif args.platform and args.nodes:
        config = PipelineConfig.from_args(args)
    else:
        parser.error('Either --config or (--platform + --nodes) is required')

    # Override output settings from CLI
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.chart_format:
        config.chart_format = args.chart_format
    if args.chart_dpi:
        config.chart_dpi = args.chart_dpi
    if args.notebook_name:
        config.notebook_name = args.notebook_name

    # Determine steps
    steps = args.step if args.step else STEPS

    # If local-dir is provided, set it on all experiments
    local_dir = args.local_dir
    if local_dir:
        for exp in config.experiments:
            if not exp.local_data_dir:
                exp.local_data_dir = local_dir

    run_pipeline(config, steps, local_dir=local_dir)


if __name__ == '__main__':
    main()