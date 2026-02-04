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
AWS Pricing module for cost tracking in serverless experiments.

Supports configurable pricing with the following precedence:
    CLI args > Environment vars > Config file > Dynamic API > Hardcoded defaults
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AWSPricing:
    """AWS pricing configuration with defaults (us-east-1, Jan 2025)"""

    # Lambda pricing (per GB-second and per request)
    lambda_gb_second: float = 0.0000166667
    lambda_request: float = 0.0000002

    # Step Functions pricing (per state transition)
    step_fn_transition: float = 0.000025

    # S3 pricing
    s3_put_request: float = 0.000005
    s3_get_request: float = 0.0000004
    s3_storage_gb_month: float = 0.023
    s3_transfer_out_gb: float = 0.09

    # EC2 pricing (for comparison, m3.xlarge on-demand)
    ec2_m3_xlarge_hourly: float = 0.266

    # ElastiCache/Redis pricing (cache.t3.micro)
    redis_node_hourly: float = 0.017

    # Metadata
    region: str = "us-east-1"
    effective_date: str = "2025-01-01"

    @classmethod
    def load(cls, config_file: Optional[str] = None, fetch_dynamic: bool = False) -> "AWSPricing":
        """
        Load pricing with precedence: env vars > config file > dynamic API > defaults

        Args:
            config_file: Path to JSON config file with pricing overrides
            fetch_dynamic: If True, attempt to fetch from AWS Price List API

        Returns:
            AWSPricing instance with configured values
        """
        pricing = cls()

        if fetch_dynamic:
            try:
                api_pricing = cls._fetch_from_aws_api()
                if api_pricing:
                    pricing = api_pricing
                    logger.info("Loaded pricing from AWS Price List API")
            except Exception as e:
                logger.debug(f"Could not fetch dynamic pricing: {e}")

        if config_file and os.path.exists(config_file):
            try:
                pricing = cls._load_from_file(config_file, pricing)
                logger.info(f"Loaded pricing from config file: {config_file}")
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")

        pricing = cls._apply_env_overrides(pricing)
        return pricing

    @classmethod
    def _load_from_file(cls, config_file: str, base: "AWSPricing") -> "AWSPricing":
        """Load pricing from JSON config file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        values = asdict(base)
        values.update(config)
        return cls(**{k: v for k, v in values.items() if k in cls.__dataclass_fields__})

    @classmethod
    def _apply_env_overrides(cls, pricing: "AWSPricing") -> "AWSPricing":
        """Apply environment variable overrides"""
        env_mapping = {
            'AWS_PRICING_LAMBDA_GB_SECOND': 'lambda_gb_second',
            'AWS_PRICING_LAMBDA_REQUEST': 'lambda_request',
            'AWS_PRICING_STEP_FN_TRANSITION': 'step_fn_transition',
            'AWS_PRICING_S3_PUT_REQUEST': 's3_put_request',
            'AWS_PRICING_S3_GET_REQUEST': 's3_get_request',
            'AWS_PRICING_S3_TRANSFER_OUT_GB': 's3_transfer_out_gb',
            'AWS_PRICING_REGION': 'region',
        }
        values = asdict(pricing)
        for env_var, field_name in env_mapping.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                field_type = cls.__dataclass_fields__[field_name].type
                if field_type == float:
                    values[field_name] = float(env_value)
                else:
                    values[field_name] = env_value
        return cls(**values)

    @classmethod
    def _fetch_from_aws_api(cls) -> Optional["AWSPricing"]:
        """Fetch pricing from AWS Price List API"""
        try:
            import boto3
            pricing_client = boto3.client('pricing', region_name='us-east-1')
            response = pricing_client.get_products(
                ServiceCode='AWSLambda',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': 'US East (N. Virginia)'},
                    {'Type': 'TERM_MATCH', 'Field': 'group', 'Value': 'AWS-Lambda-Duration'},
                ],
                MaxResults=10
            )
            for price_item in response.get('PriceList', []):
                item = json.loads(price_item)
                terms = item.get('terms', {}).get('OnDemand', {})
                for term_value in terms.values():
                    for dim_value in term_value.get('priceDimensions', {}).values():
                        price = dim_value.get('pricePerUnit', {}).get('USD')
                        if price:
                            return cls(lambda_gb_second=float(price))
        except Exception:
            pass
        return None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostMetrics:
    """Metrics captured for cost calculation"""

    # Lambda metrics
    lambda_memory_mb: int = 0
    lambda_duration_ms: float = 0.0
    lambda_invocations: int = 0

    # Step Functions metrics
    step_fn_transitions: int = 0

    # S3 metrics
    s3_put_count: int = 0
    s3_get_count: int = 0
    s3_transfer_bytes: int = 0

    # Computed costs
    lambda_cost_usd: float = 0.0
    step_fn_cost_usd: float = 0.0
    s3_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    def calculate_costs(self, pricing: AWSPricing) -> "CostMetrics":
        """Calculate costs based on metrics and pricing"""
        gb_seconds = (self.lambda_memory_mb / 1024.0) * (self.lambda_duration_ms / 1000.0)
        self.lambda_cost_usd = (
            gb_seconds * pricing.lambda_gb_second +
            self.lambda_invocations * pricing.lambda_request
        )
        self.step_fn_cost_usd = self.step_fn_transitions * pricing.step_fn_transition
        self.s3_cost_usd = (
            self.s3_put_count * pricing.s3_put_request +
            self.s3_get_count * pricing.s3_get_request +
            (self.s3_transfer_bytes / (1024**3)) * pricing.s3_transfer_out_gb
        )
        self.total_cost_usd = self.lambda_cost_usd + self.step_fn_cost_usd + self.s3_cost_usd
        return self

    def to_dict(self) -> dict:
        return asdict(self)


class CostTracker:
    """Track costs during experiment execution"""

    def __init__(self, pricing: Optional[AWSPricing] = None, pricing_config: Optional[str] = None):
        """
        Initialize cost tracker.

        Args:
            pricing: AWSPricing instance (loads defaults if None)
            pricing_config: Path to pricing config JSON file
        """
        self.pricing = pricing or AWSPricing.load(config_file=pricing_config)
        self.metrics = CostMetrics()

        # Try to get Lambda memory from environment
        lambda_memory = os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE')
        if lambda_memory:
            self.metrics.lambda_memory_mb = int(lambda_memory)

    def set_lambda_memory(self, memory_mb: int):
        """Set Lambda memory size in MB"""
        self.metrics.lambda_memory_mb = memory_mb

    def set_world_size(self, world_size: int):
        """Set world size (number of Lambda invocations)"""
        self.metrics.lambda_invocations = world_size
        # Step Functions: world_size map iterations + ~4 orchestration states
        self.metrics.step_fn_transitions = world_size + 4

    def record_duration(self, duration_ms: float):
        """Record Lambda execution duration in milliseconds"""
        self.metrics.lambda_duration_ms = duration_ms

    def record_s3_put(self, count: int = 1):
        """Record S3 PUT operations"""
        self.metrics.s3_put_count += count

    def record_s3_get(self, count: int = 1):
        """Record S3 GET operations"""
        self.metrics.s3_get_count += count

    def calculate(self) -> CostMetrics:
        """Calculate final costs and return metrics"""
        return self.metrics.calculate_costs(self.pricing)

    def get_summary_dict(self) -> dict:
        """Get summary dictionary for CSV export"""
        self.calculate()
        gb_seconds = (self.metrics.lambda_memory_mb / 1024.0) * (self.metrics.lambda_duration_ms / 1000.0)
        return {
            'lambda_memory_mb': self.metrics.lambda_memory_mb,
            'lambda_duration_ms': self.metrics.lambda_duration_ms,
            'lambda_invocations': self.metrics.lambda_invocations,
            'lambda_gb_seconds': gb_seconds,
            'lambda_cost_usd': self.metrics.lambda_cost_usd,
            'step_fn_transitions': self.metrics.step_fn_transitions,
            'step_fn_cost_usd': self.metrics.step_fn_cost_usd,
            's3_put_count': self.metrics.s3_put_count,
            's3_get_count': self.metrics.s3_get_count,
            's3_cost_usd': self.metrics.s3_cost_usd,
            'total_cost_usd': self.metrics.total_cost_usd,
        }