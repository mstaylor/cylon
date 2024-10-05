#!/bin/bash --login

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate cylon_dev

# Re-enable strict mode:
set -euo pipefail

export LD_LIBRARY_PATH=/cylon/install/lib

exec opentelemetry-instrument python /cylon/target/aws/scripts/ecs/S3_run_script.py