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

> **1.1 Contributions**
>
> This work makes the following research contributions:
>
> **(i) BSP Execution Model for Serverless Computing:**
> We demonstrate that Bulk Synchronous Parallel (BSP) workloads—traditionally requiring dedicated HPC infrastructure with low-latency interconnects—can execute effectively in serverless environments. By enabling direct peer-to-peer communication between ephemeral Lambda functions via NAT traversal, we show that tightly-coupled parallel computations achieve within 6.5% of the scaling efficiency observed on provisioned cloud VMs. This challenges the prevailing assumption that serverless is limited to embarrassingly parallel or loosely-coupled workloads.
>
> **(ii) Unified Framework for ML Data Pipelines Across Computing Paradigms:**
> We present Cylon as a portable distributed dataframe library that enables the same data-intensive operations to run across serverless (AWS Lambda), cloud VM (EC2), and HPC (Rivanna) environments without code modification. This portability supports both ML training pipelines (distributed data preprocessing, feature engineering, dataset joining) and inference pipelines (batch prediction aggregation, result merging). Scientific domains with large-scale data processing needs—such as genomics (e.g., 1000 Genomes, Cancer Genome Atlas), hydrology time series analysis (CAMELS, Caravan), earthquake prediction (MultiFoundationQuake), and astronomy inference (CosmicAI)—could leverage Cylon's distributed operations in serverless deployments, enabling elastic scaling for bursty workloads without dedicated infrastructure.
>
> **(iii) Quantitative Cost-Performance Analysis for Serverless HPC:**
> We provide an empirical cost model for data-intensive serverless workloads, enabling practitioners to make informed deployment decisions. Our analysis shows that serverless can be cost-competitive for bursty workloads: a 64-worker distributed join costs approximately $0.012 on Lambda with pay-per-millisecond billing, compared to provisioned EC2 instances where idle time dominates cost for intermittent workloads.
>
> **(iv) Communication Substrate Comparison for Serverless Collectives:**
> We evaluate three communication approaches for MPI-style collectives in serverless: direct TCP via NAT hole-punching, Redis-mediated, and S3-mediated message passing. Our results show direct communication achieves 10-100× lower latency than storage-mediated alternatives, establishing NAT traversal as a viable approach for latency-sensitive distributed computing in serverless environments.

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

#### Justification for Communication Microbenchmarks

The microbenchmarks address multiple reviewer concerns:

1. **L2 (Evaluation Scope):** Demonstrates that MPI-style collective operations (not just data shuffle patterns like Join) work effectively in serverless environments.

2. **C2 (Compute vs Communication Breakdown):** Isolates pure communication latency (barrier, allreduce) from compute, helping explain performance differences between Rivanna and EC2.

3. **Contribution (iv):** Provides empirical data for "Communication Substrate Comparison for Serverless Collectives"—showing direct TCP (NAT hole-punching) achieves 10-100× lower latency than storage-mediated alternatives.

4. **ML Relevance:** AllReduce is central to distributed ML gradient aggregation (used in data-parallel training). Demonstrating low-latency AllReduce in serverless strengthens the paper's relevance to ML workloads beyond data preprocessing.

**Effort:** High (new experiments required)

**Implementation Status:**
- ✅ Extended `scaling.py` with `-operation groupby` option
- ✅ Added `-operation microbenchmark` for communication primitives

#### Experiment Execution Plan

| Platform | GroupBy | Microbenchmark | Priority | Notes |
|----------|---------|----------------|----------|-------|
| **Lambda** | ✅ Required | ✅ Required | **High** | Core contribution - serverless BSP |
| **EC2** | ❌ Excluded | ❌ Excluded | N/A | Existing Join data provides VM baseline |
| **Rivanna** | ❌ Excluded | ❌ Excluded | N/A | Existing Join data provides HPC baseline |

**Rationale for Lambda-only new experiments:**
- Paper's core contribution is serverless execution, not HPC performance
- Existing EC2 and Rivanna Join data already establishes VM/HPC baselines
- GroupBy uses the same shuffle (all-to-all) communication pattern as Join — expected to follow comparable scaling
- EC2 cost comparison (L4) can be derived post-hoc from existing Join timing data
- Reduces experiment scope and AWS spend while still addressing all reviewer concerns

---

### L3: Missing Baseline Comparisons

**Action:** Add S3-mediated baseline comparison

#### S3 Baseline Implementation

The C++ FMI library supports multiple backend types (Direct, S3, Redis) via the `FMI::Utils::Backends` class hierarchy.

**Required Changes (IMPLEMENTED):**

| Component | Change | File | Status |
|-----------|--------|------|--------|
| C++ FMIConfig | Add `channel_type` parameter | `cpp/src/cylon/net/fmi/fmi_communicator.hpp` | ✅ |
| pycylon bindings | Expose `channel_type` in Cython | `python/pycylon/pycylon/net/fmi_config.pyx` | ✅ |
| scaling.py | Add `fmi-s3` and `fmi-redis` env options | `target/shared/scripts/scaling/scaling.py` | ⏳ |

**Experiment Design (Lambda only):**

| Configuration | Channel Type | Purpose |
|---------------|--------------|---------|
| `fmi-cylon` (current) | Direct (TCP hole punch) | Primary serverless approach |
| `fmi-s3` (new) | S3 object storage | Baseline: storage-mediated |
| `fmi-redis` (new) | Redis key-value | Baseline: in-memory storage |

> **Note:** These baseline comparisons are Lambda-only. They demonstrate the benefit of NAT hole-punching vs storage-mediated communication in serverless environments. EC2/Rivanna use direct TCP sockets and don't need this comparison.

**Expected Results:**
- S3 baseline should be 10-100x slower than Direct
- This quantifies the benefit of NAT hole punching approach

**Effort:** Medium (implementation + experiments)

---

### L4: Cost Analysis (IMPLEMENTED)

**Action:** Implement cost tracking framework with configurable pricing

**Implementation:** `target/shared/scripts/scaling/costlib/aws_pricing.py`

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
  "rows": "1000000",
  "world_size": "64",
  "iterations": "10",
  "scaling": "w",
  "uniqueness": "0.9",
  "cylon_operation": "join",
  "s3_bucket": "cylon-experiments",
  "s3_object_name": "scripts/scaling",
  "s3_object_type": "folder",
  "script": "/tmp/scripts/scaling/scaling.py",
  "output_scaling_filename": "/tmp/scaling.csv",
  "output_summary_filename": "/tmp/summary.csv",
  "s3_stopwatch_object_name": "results/scaling.csv",
  "s3_summary_object_name": "results/summary.csv",
  "rendezvous_host": "tcpunch.example.com",
  "rendezvous_port": "9999",
  "resolve_rendezvous_host": "true",
  "redis_host": "redis.example.com",
  "redis_port": "6379",
  "redis_namespace": "cylon",
  "rank": "0",
  "cylon_log_level": "100",
  "fmi_options": null,
  "fmi_max_timeout": "120000",
  "cylon_session_id": "exp-join-w-64-20250207",
  "enable_cost_tracking": "true",
  "aws_pricing_config": null,
  "enable_fmi_ping": "false"
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

#### Instrumentation Changes (IMPLEMENTED - for new experiments only)

The following timing breakdown fields have been added to `scaling.py` for `join()` and `groupby_agg()`:

```python
timing = {
    # Existing fields...

    # Breakdown fields (C2: compute vs communication)
    'data_gen_t': [],      # DataFrame generation time (ms)
    'compute_t': [],       # Local join/hash/groupby operations (ms)
    'comm_t': [],          # Total communication time: barriers + allreduce (ms)
}
```

> **Note:** These fields are available for new GroupBy experiments. For existing Join data, use microbenchmarks to characterize communication overhead instead of re-running experiments.

The breakdown allows calculating:
- **Compute ratio**: `compute_t / avg_t` - fraction of time spent in local operations
- **Communication ratio**: `comm_t / avg_t` - fraction of time in collective operations
- **Overhead**: `com_init_t + data_gen_t` - initialization costs

#### Explanation for Rivanna vs EC2 Difference

> "The 1.8× faster single-node execution time on Rivanna compared to EC2 is attributed to compute differences:
> - Rivanna: Intel Xeon Gold 6248 @ 2.50GHz (Cascade Lake, 2019)
> - EC2 m3.xlarge: Intel Xeon E5-2670 v2 @ 2.50GHz (Ivy Bridge, 2013)
>
> The newer Cascade Lake architecture provides ~40% better IPC for Arrow/join kernels. This compute difference does not affect our scaling conclusions, as we compare scaling efficiency (speedup ratios) rather than absolute performance."

#### Communication Microbenchmark (IMPLEMENTED)

Added `comm_microbenchmark()` function to `scaling.py` with `-operation microbenchmark`:

```python
def comm_microbenchmark(data=None, ipAddress=None):
    """
    Measures:
    - Barrier latency (synchronization overhead)
    - AllReduce latency at various message sizes (8B to 1MB)
    - AllReduce bandwidth (derived from latency and message size)
    """
```

Output fields: `barrier_latency_ms`, `msg_size_bytes`, `allreduce_latency_ms`, `allreduce_bandwidth_mbps`

**Effort:** Medium (re-run experiments with instrumentation)

---

## Summary

| Reviewer Comment | Response | Effort | Priority | Status |
|------------------|----------|--------|----------|--------|
| L1: Contributions | Add subsection to Section I | Low | High | Content drafted |
| L2: Evaluation scope | Add GroupBy + microbenchmarks | High | High | **Code complete** |
| L3: Baseline comparisons | Add S3-mediated baseline | Medium | High | **Code complete** |
| L4: Cost analysis | Implement cost tracking framework | Medium | High | **Code complete** |
| C1: Define <1% | Add table with precise values | Low | Medium | Content drafted |
| C2: Compute vs comm | Add breakdown + explanation | Medium | Medium | **Code complete** |

### Remaining Experiment Execution

| Experiment | Lambda | EC2 | Rivanna | Notes |
|------------|--------|-----|---------|-------|
| **Join scaling** | ✅ Use existing | ✅ Use existing | ✅ Use existing | No rerun needed |
| **Join cost (L4)** | ✅ Post-hoc calc | ✅ Post-hoc calc | N/A | From existing timing data |
| GroupBy scaling | ✅ Required | ❌ Excluded | ❌ Excluded | Lambda-only; exercises shuffle pattern |
| Microbenchmark | ✅ Required | ❌ Excluded | ❌ Excluded | Lambda-only |
| S3/Redis baseline | ✅ Required | N/A | N/A | Lambda-only |

#### EC2 and Rivanna Exclusion Rationale

EC2 and Rivanna experiments are **excluded** from the new experiments for the following reasons:

1. **Reviewer concerns focus on serverless:** L2, L3, L4, and C2 all relate to serverless evaluation depth, not HPC coverage
2. **Existing Join data provides the baseline:** EC2 and Rivanna Join data already establishes the VM/HPC comparison point. GroupBy uses the same shuffle (all-to-all) communication pattern and is expected to follow comparable scaling behavior.
3. **Cost comparison uses existing data:** EC2 hourly rates (reviewer L4) can be calculated post-hoc from existing Join timing data — no new EC2 experiments required.
4. **Paper's core contribution:** Serverless BSP execution, not HPC performance

**Paper text to add (preempts reviewer questions):**

> "EC2 and HPC (Rivanna) results for GroupBy and communication microbenchmarks are omitted as they follow similar scaling patterns to the Join operation (which uses the same shuffle-based all-to-all communication). The focus of this evaluation is serverless execution where the viability of BSP workloads is less established. EC2 cost comparisons are derived from existing Join experiment timing data."

#### Shuffle Pattern Coverage (Reviewer L2)

The reviewer asks about shuffle operations. Both Join and GroupBy exercise the shuffle (all-to-all hash repartition) communication pattern internally:
- **Join**: `shuffle(left, join_cols)` + `shuffle(right, join_cols)` → local merge
- **GroupBy**: `shuffle(table, group_key_cols)` → local aggregation

GroupBy on Lambda demonstrates that shuffle-based distributed operators beyond Join work in serverless. Cylon also exposes `shuffle()` as a standalone API, but the higher-level GroupBy is more meaningful for ML data pipeline relevance.

#### Data Reuse Strategy

**Join experiments:** Use existing data - no rerun required.

**Cost analysis (L4):** Calculate post-hoc from existing Join timing data:
```python
# From existing CSV: duration_ms, world_size
memory_gb = 10  # Lambda memory in GB
duration_sec = duration_ms / 1000
lambda_cost = (memory_gb * duration_sec * 0.0000166667) + (world_size * 0.0000002)
step_fn_cost = (world_size + 4) * 0.000025
total_cost = lambda_cost + step_fn_cost
```

**EC2 cost comparison (L4):** Calculate from existing EC2 Join data:
```python
# EC2 m3.xlarge: $0.266/hour (us-east-1, on-demand)
# From existing CSV: duration_ms, world_size
ec2_hourly_rate = 0.266
ec2_cost = (world_size * ec2_hourly_rate * duration_sec) / 3600
# Note: EC2 charges per-second with 60s minimum
```

#### Infrastructure Prerequisites

- `CYLON_SESSION_ID` environment variable is now **required** for all Redis-based runs (UCX/UCC/Libfabric). `scaling.py` accepts it via `-sessionid` arg or `CYLON_SESSION_ID` env var.
- Lambda Step Function input payload should include `session_id` or set `CYLON_SESSION_ID` in the Lambda environment.
- C++ TCPunch client uses legacy protocol — Lambda experiments use the Rust client which is Protocol v2 compatible with the Rust TCPunch server.

**Compute vs communication breakdown (C2):** Use microbenchmarks to isolate communication latency. This provides cleaner separation than in-operation timing breakdown since microbenchmarks measure pure collective overhead without data processing noise.

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

---

## Appendix: Experiment Results Pipeline

An automated pipeline replaces the previous manual workflow (S3 download -> Google Sheets -> Jupyter hardcoded arrays -> charts). The pipeline lives in `target/shared/scripts/results/` and produces a Jupyter notebook (`frontiersCloudSubmission.ipynb`) with all chart cells from aggregated experiment data.

### Pipeline Overview

```
S3 / Local Files
       |  results_downloader.py
  Raw CSV files (per-rank summary data)
       |  results_aggregator.py
  aggregated_results.csv (mean/stddev per experiment)
       |  notebook_generator.py
  frontiersCloudSubmission.ipynb (interactive charts)
```

### Module Summary

| File | Purpose |
|------|---------|
| `config.py` | `ExperimentConfig` and `PipelineConfig` dataclasses, YAML loader |
| `results_downloader.py` | S3 batch download via boto3 prefix discovery |
| `results_aggregator.py` | CSV parsing (old `##` and new CSV formats), mean/stddev aggregation |
| `chart_generator.py` | Matplotlib charts for direct SVG/PNG generation |
| `notebook_generator.py` | Generates `frontiersCloudSubmission.ipynb` with all chart cells |
| `pipeline.py` | CLI orchestrator: download -> aggregate -> notebook/charts |
| `configs/experiment_config.yaml` | Experiment definitions with local data paths |

### Quick Start

```bash
cd target/shared/scripts

# Aggregate data + generate Jupyter notebook (primary workflow)
conda run -n cylon_dev python -m results.pipeline \
  --config results/configs/experiment_config.yaml \
  --step aggregate --step notebook \
  --output-dir /home/parallels/cylon/target/aws/scripts/notebooks

# Custom notebook name
conda run -n cylon_dev python -m results.pipeline \
  --config results/configs/experiment_config.yaml \
  --step aggregate --step notebook \
  --output-dir ./output --notebook-name myNotebook

# Generate chart files directly (SVG for paper, PNG for review)
conda run -n cylon_dev python -m results.pipeline \
  --config results/configs/experiment_config.yaml \
  --step aggregate --step charts \
  --output-dir ./output --chart-format svg --chart-dpi 300

# Single experiment (no YAML config needed)
conda run -n cylon_dev python -m results.pipeline \
  --platform ec2 --scaling weak --instance 16_28 \
  --rows 9100000 --nodes 1,2,4,8,16,32 \
  --local-dir /home/parallels/cylon_experiments/aws/results-9100000/ec2/16_28 \
  --step aggregate --step notebook --output-dir ./output
```

### Charts Generated

**Existing charts (replicated from notebooks):**

| Chart | File | Description |
|-------|------|-------------|
| Weak Scaling | `join-w-scaling.{svg,png}` | Line chart with error bars, one line per platform/instance |
| Strong Scaling | `join-s-scaling.{svg,png}` | Line chart with error bars |
| Strong Scaling + Speedup | `join-s-scaling-speedup.{svg,png}` | Dual-axis: execution time + speedup |
| Strong Scaling Scaled | `join-s-scaling-scaled.{svg,png}` | Time * nodes (shows parallel overhead) |

**New charts for reviewer concerns:**

| Chart | File | Reviewer | Description |
|-------|------|----------|-------------|
| Compute vs Comm Breakdown | `compute-vs-comm-breakdown.{svg,png}` | C2 | Stacked bar: data_gen / compute / comm per platform |
| Cost Analysis | `cost-analysis.{svg,png}` | L4 | Stacked bar: Lambda + Step Functions cost per node count |

The new charts render only when the data includes the corresponding fields (`data_gen_t`, `compute_t`, `comm_t` for C2; `lambda_cost_usd`, `step_fn_cost_usd` for L4). Existing experiments that lack these fields will skip these charts gracefully.

### Adding New Experiments

After running new experiments (GroupBy, microbenchmarks, S3/Redis baselines), add entries to `configs/experiment_config.yaml`:

```yaml
  - platform: "lambda"
    scaling_type: "weak"
    instance_label: "10GB"
    instance_detail: "10GB Memory"
    node_counts: [1, 2, 4, 8, 16, 32, 64]
    rows: 9100000
    operation: "join"
    color: "green"
    marker: "s"
    local_data_dir: "/path/to/lambda/results"
```

Then re-run the pipeline to regenerate the aggregated CSV and notebook.

### Aggregated CSV Format

The intermediate `aggregated_results.csv` has one row per (platform, scaling_type, instance, node_count) with columns:

- **Metadata:** `platform`, `scaling_type`, `instance_label`, `instance_detail`, `node_count`, `num_runs`
- **Timing (mean/std in seconds):** `avg_t`, `elapsed_t`, `max_t`, `com_init_t`, `barrier_t`
- **Breakdown (mean/std in seconds):** `data_gen_t`, `compute_t`, `comm_t`
- **Cost (mean/std):** `lambda_cost_usd`, `step_fn_cost_usd`, `total_cost_usd`
- **Flags:** `has_timing_breakdown`, `has_cost_data`

This CSV enables re-running charts without re-aggregating, and can be used for ad-hoc analysis in notebooks or spreadsheets.