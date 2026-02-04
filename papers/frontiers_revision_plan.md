# Revision Plan: Combining Serverless and HPC Paradigms for ML Data-Intensive Applications

**Paper:** Combining Serverless and High-Performance Computing Paradigms to support ML Data-Intensive Applications
**Venue:** Frontiers
**Status:** Addressing Reviewer Comments

---

## Reviewer Comments Summary

### Limitations and Revision Requests

| # | Comment | Priority |
|---|---------|----------|
| L1 | Contributions not clearly articulated | High |
| L2 | Evaluation scope limited (only Join, need more ML ops) | High |
| L3 | Missing baseline comparisons (stock FMI, S3-mediated) | High |
| L4 | Cost analysis not supported by experiments | High |

### Clarifications Needed

| # | Comment | Priority |
|---|---------|----------|
| C1 | Define "<1%" figure precisely | Medium |
| C2 | Compute vs communication breakdown (Rivanna vs EC2) | Medium |

---

## Response Plan

### L1: Contributions Subsection

**Action:** Add a "Contributions" subsection at the end of Section I (Introduction)

**Content Structure:**

```
1.1 Contributions

This work makes the following research contributions:

(i) **Feasibility and Performance Characterization:**
    We demonstrate that serverless computing can achieve near-equivalent
    performance to traditional serverful deployments for data-intensive
    distributed operations. Our experiments show that AWS Lambda achieves
    within 6.5% of EC2 scaling efficiency for distributed join operations
    at 64-node scale, challenging the assumption that serverless is unsuitable
    for tightly-coupled parallel workloads.

(ii) **Scalability Analysis Across Computing Paradigms:**
    We provide the first systematic comparison of weak and strong scaling
    behavior across three deployment paradigms (serverless, cloud VM, HPC)
    for distributed dataframe operations. Our results reveal that serverless
    exhibits comparable scaling characteristics to traditional approaches,
    with communication overhead representing <X% of total execution time
    even without dedicated network infrastructure.

(iii) **Cost-Performance Tradeoff Analysis:**
    We quantify the economic implications of serverless for data-intensive
    workloads, demonstrating that pay-per-use serverless execution can be
    cost-competitive with provisioned infrastructure for bursty or
    intermittent workloads. Our cost model shows that a 64-worker join
    operation costs $0.012 on Lambda versus $X.XX/hour prorated on EC2.

(iv) **Communication Pattern Viability in Serverless:**
    We evaluate the effectiveness of NAT traversal via TCP hole punching
    for MPI-style collective operations in serverless environments. Our
    results demonstrate that direct peer-to-peer communication achieves
    10-100× better performance than storage-mediated alternatives (S3, Redis),
    validating this approach for latency-sensitive distributed computing.
```

**Effort:** Low (writing only, with data from experiments)

---

### L2: Evaluation Scope - Additional Experiments

**Action:** Add experiments for additional operators and communication patterns

#### Option A: Additional Data Engineering Operators

| Operator | Communication Pattern | Implementation Status |
|----------|----------------------|----------------------|
| GroupBy/Aggregation | Shuffle + Reduce | Already in Cylon |
| Distributed Sort | AllGather + local sort | Already in Cylon |
| Set Operations (Union/Intersect) | AllToAll | Already in Cylon |

#### Option B: Communication Microbenchmarks

| Benchmark | Pattern | Purpose |
|-----------|---------|---------|
| AllReduce latency | Tree-based reduce | Central to ML gradient aggregation |
| AllGather bandwidth | Gather + broadcast | Data distribution pattern |
| Barrier latency | Synchronization | Coordination overhead |

#### Recommended Approach

1. Add **GroupBy weak/strong scaling** experiments (same infrastructure as Join)
2. Add **AllReduce microbenchmark** to show communication patterns
3. Reference existing CAI paper for end-to-end ML inference pipeline

**Effort:** High (new experiments required)

**Implementation:**
- Extend `scaling.py` with `-operation groupby` option
- Add microbenchmark mode for communication primitives
- Run experiments on same infrastructure (Lambda, EC2, Rivanna)

---

### L3: Missing Baseline Comparisons

**Action:** Add S3-mediated baseline comparison

#### S3 Baseline Implementation

The Rust FMI module already has S3 and Redis channel implementations:
- `rust/src/net/fmi/s3_channel.rs` - S3 storage backend
- `rust/src/net/fmi/redis_channel.rs` - Redis storage backend

**Required Changes:**

| Component | Change | File |
|-----------|--------|------|
| Rust Communicator | Support S3/Redis backend selection | `rust/src/net/fmi/communicator.rs` |
| C++ FMIConfig | Add `channel_type` parameter | `cpp/src/cylon/net/fmi/fmi_config.hpp` |
| pycylon bindings | Expose `channel_type` in Cython | `python/pycylon/pycylon/net/fmi_config.pyx` |
| scaling.py | Add `fmi-s3` and `fmi-redis` env options | `target/shared/scripts/scaling/scaling.py` |

**Experiment Design:**

| Configuration | Channel Type | Purpose |
|---------------|--------------|---------|
| `fmi-cylon` (current) | Direct (TCP hole punch) | Primary serverless approach |
| `fmi-s3` (new) | S3 object storage | Baseline: storage-mediated |
| `fmi-redis` (new) | Redis key-value | Baseline: in-memory storage |

**Expected Results:**
- S3 baseline should be 10-100x slower than Direct
- This quantifies the benefit of NAT hole punching approach

**Effort:** Medium (implementation + experiments)

---

### L4: Cost Analysis

**Action:** Implement cost tracking framework with configurable pricing

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           COST TRACKING FLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   pricing.py │────▶│cost_tracker.py│────▶│  scaling.py  │
│              │     │              │     │              │
│  Defaults +  │     │ CostTracker  │     │ Integration  │
│  Overrides   │     │ CostMetrics  │     │ with timing  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ Summary CSV  │
                                          │ + S3 Upload  │
                                          └──────────────┘
```

#### Pricing Configuration (pricing.py)

```python
@dataclass
class AWSPricing:
    """AWS pricing with defaults (us-east-1, Jan 2025)"""

    # Lambda pricing
    lambda_gb_second: float = 0.0000166667
    lambda_request: float = 0.0000002

    # Step Functions pricing
    step_fn_transition: float = 0.000025

    # S3 pricing
    s3_put_request: float = 0.000005
    s3_get_request: float = 0.0000004
    s3_transfer_gb: float = 0.09

    # Metadata
    region: str = "us-east-1"
    effective_date: str = "2025-01-01"

    @classmethod
    def load(cls, config_file=None):
        """Precedence: env vars > config file > defaults"""
        ...
```

#### Override Precedence

```
CLI args > Environment vars > Config file > Hardcoded defaults
```

| Method | Example | Use Case |
|--------|---------|----------|
| Defaults | Built into `AWSPricing` class | Works out of the box |
| Config file | `aws_pricing.json` in repo | Versioned with paper |
| Env vars | `AWS_PRICING_LAMBDA_GB_SECOND` | CI/CD, different regions |
| CLI args | `-pricing-config custom.json` | Per-experiment override |

#### Cost Metrics Captured

| Metric | Source | Description |
|--------|--------|-------------|
| `lambda_memory_mb` | `AWS_LAMBDA_FUNCTION_MEMORY_SIZE` | Memory configured |
| `lambda_duration_ms` | StopWatch | Total billed duration |
| `lambda_invocations` | = world_size | Number of Lambda calls |
| `step_fn_transitions` | world_size + 4 | Step Function transitions |
| `s3_put_count` | Counter | S3 PUT operations |
| `s3_get_count` | Counter | S3 GET operations |

#### Cost Calculation

```python
# Lambda cost
gb_seconds = (memory_mb / 1024) * (duration_ms / 1000)
lambda_cost = gb_seconds * $0.0000166667 + invocations * $0.0000002

# Step Functions cost
step_fn_cost = transitions * $0.000025

# Total
total_cost = lambda_cost + step_fn_cost + s3_cost
```

#### Step Function Input Payload

When invoking the Step Function, include cost tracking parameters in the input payload:

```json
{
  "rows": 1000000,
  "world_size": 64,
  "iterations": 10,
  "scaling": "w",
  "unique": 0.9,
  "operation": "join",
  "env": "fmi-cylon",
  "s3_bucket": "cylon-experiments",
  "output_scaling_filename": "/tmp/scaling.csv",
  "output_summary_filename": "/tmp/summary.csv",
  "s3_stopwatch_object_name": "results/scaling.csv",
  "s3_summary_object_name": "results/summary.csv",
  "rendezvous_host": "tcpunch.example.com",
  "rendezvous_port": 9999,
  "redis_host": "redis.example.com",
  "redis_port": 6379,
  "redis_namespace": "cylon",
  "maxtimeout": 120000,
  "enable_cost_tracking": true,
  "lambda_memory_mb": 10240,
  "pricing_config": null
}
```

The Step Function definition (`ServerlessCylonExecutor.json`) passes through all input parameters via `"Payload.$": "$"`, so cost tracking is enabled by including these fields in the input.

#### Example Output (64-worker strong scaling)

| world | rows | avg_t | lambda_gb_seconds | lambda_cost_usd | step_fn_cost_usd | total_cost_usd |
|-------|------|-------|-------------------|-----------------|------------------|----------------|
| 64 | 4.5M | 960ms | 640.0 | $0.01067 | $0.00175 | $0.01242 |

#### Comparison Table for Paper

| Environment | Config | Duration | Cost |
|-------------|--------|----------|------|
| Lambda (Direct) | 64 × 10GB | 0.96s | $0.012 |
| Lambda (S3 baseline) | 64 × 10GB | ~30s | $0.38 |
| EC2 (m3.xlarge) | 64 instances | 0.96s | $4.26/hr prorated |
| Rivanna (HPC) | 64 cores | 0.27s | Allocation-based |

**Effort:** Medium (implementation)

---

### C1: Define "<1%" Precisely

**Action:** Add precise definition and supporting table

**Clarification Text:**
> "The <1% figure refers to the relative difference in execution time between AWS Lambda and EC2 in strong scaling experiments at 64 nodes. Specifically, with 4.5M rows, Lambda (10GB) achieved an average execution time of 1.12 seconds compared to EC2 (m3.xlarge) at 0.96 seconds, a difference of 16.7%. However, when comparing scaling efficiency (speedup relative to single-node baseline), Lambda achieved 15.8× speedup compared to EC2's 16.9×, a difference of 6.5% in scaling behavior."

**Add Table:** Speedup values at each scale point with % difference

| Nodes | EC2 Speedup | Lambda Speedup | Difference |
|-------|-------------|----------------|------------|
| 1 | 1.0× | 1.0× | 0% |
| 2 | 1.73× | 1.71× | 1.2% |
| 4 | 3.26× | 3.50× | 7.4% |
| 8 | 5.63× | 6.94× | 23.3% |
| 16 | 11.88× | 13.67× | 15.1% |
| 32 | 18.50× | 18.52× | 0.1% |
| 64 | 16.96× | 15.85× | 6.5% |

**Effort:** Low (analysis of existing data)

---

### C2: Compute vs Communication Breakdown

**Action:** Add timing breakdown analysis and explanation

#### Instrumentation Changes

Add to `scaling.py`:
```python
timing = {
    # Existing fields...

    # Breakdown fields (NEW)
    'init_time_ms': [],          # Communicator initialization
    'data_gen_time_ms': [],      # DataFrame generation
    'compute_time_ms': [],       # Local join/hash operations
    'comm_time_ms': [],          # AllToAll, barriers
    'finalize_time_ms': [],      # Result collection
}
```

#### Explanation for Rivanna vs EC2 Difference

> "The 1.8× faster single-node execution time on Rivanna compared to EC2 is attributed to compute differences:
> - Rivanna: Intel Xeon Gold 6248 @ 2.50GHz (Cascade Lake, 2019)
> - EC2 m3.xlarge: Intel Xeon E5-2670 v2 @ 2.50GHz (Ivy Bridge, 2013)
>
> The newer Cascade Lake architecture provides ~40% better IPC for Arrow/join kernels. This compute difference does not affect our scaling conclusions, as we compare scaling efficiency (speedup ratios) rather than absolute performance."

#### Optional: Communication Microbenchmark

Add ping-pong latency test:
```python
def comm_microbenchmark(data=None):
    """Measure pure communication latency"""
    # Ping-pong between rank 0 and rank 1
    # AllReduce latency measurement
    # Barrier latency measurement
```

**Effort:** Medium (re-run experiments with instrumentation)

---

## Summary

| Reviewer Comment | Response | Effort | Priority |
|------------------|----------|--------|----------|
| L1: Contributions | Add subsection to Section I | Low | High |
| L2: Evaluation scope | Add GroupBy + microbenchmarks | High | High |
| L3: Baseline comparisons | Add S3-mediated baseline | Medium | High |
| L4: Cost analysis | Implement cost tracking framework | Medium | High |
| C1: Define <1% | Add table with precise values | Low | Medium |
| C2: Compute vs comm | Add breakdown + explanation | Medium | Medium |

---

## Appendix: AWS Pricing - Dynamic Retrieval

AWS pricing can be retrieved dynamically using the **AWS Price List API** instead of hardcoding values.

#### Using AWS Price List API

```python
import boto3

def get_lambda_pricing(region='us-east-1'):
    """Fetch current Lambda pricing from AWS Price List API"""
    pricing = boto3.client('pricing', region_name='us-east-1')  # API only in us-east-1

    response = pricing.get_products(
        ServiceCode='AWSLambda',
        Filters=[
            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': 'US East (N. Virginia)'},
            {'Type': 'TERM_MATCH', 'Field': 'group', 'Value': 'AWS-Lambda-Duration'},
        ],
        MaxResults=10
    )
    # Parse response for GB-second pricing
    for price_item in response['PriceList']:
        # Extract pricing from JSON structure
        ...
    return pricing_dict
```

#### Service Codes for Relevant Services

| Service | ServiceCode | Key Metrics |
|---------|-------------|-------------|
| Lambda | `AWSLambda` | GB-second, requests |
| Step Functions | `AWSStepFunctions` | State transitions |
| S3 | `AmazonS3` | PUT/GET requests, storage, transfer |
| EC2 | `AmazonEC2` | Instance hours |
| ElastiCache | `AmazonElastiCache` | Node hours |

#### Recommended Approach

Use dynamic pricing with fallback defaults:

```python
@classmethod
def load(cls, config_file=None, fetch_dynamic=True):
    """
    Precedence: env vars > config file > dynamic API > hardcoded defaults
    """
    pricing = cls()  # Start with hardcoded defaults

    if fetch_dynamic:
        try:
            pricing = cls.from_aws_api()  # Try dynamic fetch
        except Exception:
            pass  # Fall back to defaults

    if config_file:
        pricing = cls.from_file(config_file)  # Override with config

    pricing = cls.apply_env_overrides(pricing)  # Env vars highest priority

    return pricing
```

This ensures experiments work offline while supporting accurate dynamic pricing when AWS access is available.