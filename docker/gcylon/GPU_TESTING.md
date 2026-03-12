# gcylon GPU Testing Guide

This guide covers how to test the gcylon Docker image on AWS with a GPU.

## Prerequisites

- AWS account with access to GPU instances
- Docker image: `qad5gv/gcylon:latest`

## 1. Launch AWS EC2 GPU Instance

### Recommended Instance Types

| Instance | GPU | GPU Memory | Cost |
|----------|-----|------------|------|
| g4dn.xlarge | T4 | 16 GB | Cheapest |
| g5.xlarge | A10G | 24 GB | Good balance |
| p3.2xlarge | V100 | 16 GB | High performance |

### AMI Selection

Use the **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** - it has NVIDIA drivers pre-installed.

## 2. Instance Setup

SSH into your instance and run:

```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 3. Pull the gcylon Image

```bash
sudo docker pull qad5gv/gcylon:latest
```

## 4. Verify GPU Access

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest nvidia-smi
```

Expected output should show the GPU details.

## 5. Test Cases

### 5.1 Basic CUDA Test (numba)

Tests that CUDA is accessible and the GPU is detected.

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
python -c "
import numba.cuda as cuda
print(\"CUDA available:\", cuda.is_available())
print(\"Device:\", cuda.get_current_device().name)
"'
```

**Expected:** CUDA available: True, with GPU device name displayed.

### 5.2 CuPy GPU Array Test

Tests GPU array creation and GPU-to-CPU data transfer.

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
python -c "
import cupy as cp
print(\"Creating GPU array...\")
arr = cp.array([1, 2, 3, 4, 5])
print(\"GPU array created\")
host_arr = arr.get()
print(\"Transferred to CPU:\", host_arr)
print(\"CuPy SUCCESS!\")
"'
```

**Expected:** Array created on GPU and successfully transferred to CPU.

### 5.3 cudf DataFrame Test

Tests cudf DataFrame creation and pandas conversion.

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
python -u -c "
import cudf
print(\"cudf version:\", cudf.__version__)
df = cudf.DataFrame({\"a\": [1, 2, 3], \"b\": [4, 5, 6]})
print(\"DataFrame created\")
print(\"Shape:\", df.shape)
print(\"Columns:\", df.columns.tolist())
pdf = df.to_pandas()
print(\"Converted to pandas:\")
print(pdf)
print(\"cudf SUCCESS!\")
"'
```

**Expected:** DataFrame created and converted to pandas successfully.

### 5.4 cudf Merge Test

Tests GPU-accelerated DataFrame merge operations.

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
python -u -c "
import cudf
print(\"cudf version:\", cudf.__version__)

df1 = cudf.DataFrame({\"key\": [1, 2, 3], \"val\": [10, 20, 30]})
df2 = cudf.DataFrame({\"key\": [2, 3, 4], \"val2\": [200, 300, 400]})

print(\"DataFrame 1:\")
print(df1.to_pandas())

print(\"\nDataFrame 2:\")
print(df2.to_pandas())

merged = df1.merge(df2, on=\"key\")
print(\"\nMerged (inner join on key):\")
print(merged.to_pandas())
print(\"\ncudf merge SUCCESS!\")
"'
```

**Expected:** Two DataFrames merged on the GPU successfully.

### 5.5 RMM Memory Test

Tests RAPIDS Memory Manager initialization.

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
python -c "
import rmm
print(\"RMM version:\", rmm.__version__)
rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30)
print(\"RMM pool initialized with 1GB\")
print(\"RMM SUCCESS!\")
"'
```

**Expected:** RMM memory pool initialized successfully.

## 6. gcylon Rust Examples

### 6.1 GPU Join (Single Node)

Tests distributed join on GPU with synthetic data.

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/gpu_join 10MB'
```

**Expected output:**
```
Rank 0: Using GPU 0 of 1
Rank 0: Creating tables with 4 columns, 312500 rows each
Rank 0: Left table: 312500 rows, Right table: 312500 rows
Rank 0: Distributed join (INNER) completed in Xms, output has 156250 rows, 8 columns
...
GPU DISTRIBUTED JOIN COMPLETED SUCCESSFULLY
```

### 6.2 GPU Join with Larger Data

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/gpu_join 100MB'
```

### 6.3 GPU Sort

Tests distributed sort on GPU. Requires multi-rank MPI communication.

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/gpu_sort 10MB'
```

**Note:** May fail in WSL2 environments due to GPU Direct RDMA limitations. See [WSL2 Known Issues](#wsl2-gpu-direct-rdma-limitations).

### 6.4 GPU Shuffle

Tests distributed shuffle on GPU. Requires multi-rank MPI communication.

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/gpu_shuffle 10MB'
```

**Note:** May fail in WSL2 environments due to GPU Direct RDMA limitations. See [WSL2 Known Issues](#wsl2-gpu-direct-rdma-limitations).

### 6.5 Table From Vectors Example

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/table_from_vectors_example'
```

### 6.6 GroupBy Example

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/groupby_example'
```

### 6.7 Select Example

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/select_example'
```

### 6.8 Project Example

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/project_example'
```

## 7. Interactive Testing

To run an interactive shell in the container:

```bash
sudo docker run --rm -it --gpus all qad5gv/gcylon:latest bash
```

Once inside:
```bash
# Conda environment is auto-activated
nvidia-smi
python -c "import cudf; print(cudf.__version__)"
cd /cylon/rust
./target/release/examples/gpu_join 10MB
```

## 8. Expected Test Results

### Native Linux (AWS EC2)

On AWS EC2 with native Linux and supported GPUs (T4, A10G, V100), all tests should pass:

| Test | Expected Result |
|------|-----------------|
| Basic CUDA (numba) | Pass |
| CuPy GPU Array | Pass |
| cudf DataFrame | Pass |
| cudf Merge | Pass |
| RMM Memory | Pass |
| gpu_join | Pass |
| gpu_sort | Pass |
| gpu_shuffle | Pass |
| table_from_vectors_example | Pass |
| groupby_example | Pass |
| select_example | Pass |
| project_example | Pass |

### WSL2 Environment

On WSL2, some tests will fail due to GPU Direct RDMA limitations:

| Test | Expected Result | Notes |
|------|-----------------|-------|
| Basic CUDA (numba) | Pass | |
| CuPy GPU Array | Pass | |
| cudf DataFrame | Varies | May fail on RTX 5090 (Blackwell) |
| cudf Merge | Varies | May fail on RTX 5090 (Blackwell) |
| RMM Memory | Pass | |
| gpu_join | Pass | Single-rank, no RDMA needed |
| gpu_sort | **Fail** | Requires GPU Direct RDMA |
| gpu_shuffle | **Fail** | Requires GPU Direct RDMA |
| table_from_vectors_example | Pass | |
| groupby_example | Pass | |
| select_example | Pass | |
| project_example | Pass | |

## 9. Known Issues

### WSL2 GPU Direct RDMA Limitations

**Symptom:** `gpu_sort` and `gpu_shuffle` crash with:
```
Segmentation fault: invalid permissions for mapped object at address 0xb10a00000
```

**Cause:** WSL2 uses a paravirtualized GPU driver (vGPU) that doesn't support GPU Direct RDMA (GDR). When UCX tries to do RDMA directly on GPU memory for MPI communication, the memory registration fails.

**Evidence:**
- `gpu_join` works (single rank = no inter-process RDMA needed)
- `gpu_sort`/`gpu_shuffle` fail (requires MPI all-to-all with RDMA)

**Workaround:** Disable GPU Direct RDMA by forcing TCP transport:

```bash
sudo docker run --rm --gpus all \
    -e UCX_TLS=tcp,cuda_copy \
    qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/gpu_sort 10MB'
```

**Alternative workaround:** Disable UCX CUDA support entirely:

```bash
sudo docker run --rm --gpus all \
    -e UCX_TLS=tcp,sm,self \
    qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/gpu_shuffle 10MB'
```

**Verification:** To confirm this is a WSL2 issue, run the same tests on AWS EC2 with native Linux. GPU Direct RDMA should work properly on native hardware.

### RTX 5090 (Blackwell, sm_120)

The RTX 5090 uses the new Blackwell architecture (compute capability sm_120) which may not be fully supported by RAPIDS 25.02:

- Basic GPU operations work
- `cudf.DataFrame.to_pandas()` may crash (segfault) - this is a Blackwell PTX/kernel issue, not WSL2 related
- gcylon Rust `gpu_join` works
- Distributed operations (sort, shuffle) fail in WSL2 (see above)

**Workaround for cudf issues:** Use `CUDA_LAUNCH_BLOCKING=1`:
```bash
sudo docker run --rm --gpus all -e CUDA_LAUNCH_BLOCKING=1 qad5gv/gcylon:latest ...
```

### Distinguishing WSL2 vs Blackwell Issues

| Issue | Cause | Test to Verify |
|-------|-------|----------------|
| `gpu_sort`/`gpu_shuffle` segfault in MPI/UCX | WSL2 (no GDR support) | Test on AWS EC2 - should pass |
| `cudf.to_pandas()` segfault | Blackwell (sm_120) | Test on older GPU (T4/V100) - should pass |

## 10. Versions

Current image versions:
- CUDA: 12.8.0
- cudf: 25.02.02
- RMM: 25.02.00
- UCX: 1.19.1
- UCC: 1.3.0

## 11. Troubleshooting

### Check UCX Version

```bash
sudo docker run --rm qad5gv/gcylon:latest /ucx/install/bin/ucx_info -v
```

### Check Available UCX Transports

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
ucx_info -d'
```

### Enable UCX Debug Logging

```bash
sudo docker run --rm --gpus all \
    -e UCX_LOG_LEVEL=debug \
    qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
cd /cylon/rust &&
./target/release/examples/gpu_sort 10MB'
```

### Check GPU Memory

```bash
sudo docker run --rm --gpus all qad5gv/gcylon:latest bash -c '
. /opt/conda/etc/profile.d/conda.sh && conda activate cylon_dev &&
python -c "
import cupy as cp
mempool = cp.get_default_memory_pool()
print(\"Used:\", mempool.used_bytes() / 1e9, \"GB\")
print(\"Total:\", mempool.total_bytes() / 1e9, \"GB\")
"'
```
