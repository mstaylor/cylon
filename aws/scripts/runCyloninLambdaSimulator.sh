#!/bin/bash --login

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate cylon_dev

# Re-enable strict mode:
set -euo pipefail

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

#export LD_LIBRARY_PATH=/cylon/install/lib

exec /cylon/aws/scripts/aws-lambda-rie "$@"