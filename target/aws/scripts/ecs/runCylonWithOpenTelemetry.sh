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
export OTEL_EXPORTER_OTLP_ENDPOINT=http://cylon-collector1.aws-cylondata.com:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc

exec opentelemetry-instrument python /cylon/target/aws/scripts/ecs/S3_run_script.py