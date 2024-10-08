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

# Uncomment the appropriate protocol for your programming language.
# Only for OTLP/gRPC.
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.uptrace.dev:4317"
# Only for OTLP/HTTP.
#export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.uptrace.dev"

# Pass Uptrace DSN in gRPC/HTTP headers.
export OTEL_EXPORTER_OTLP_HEADERS="uptrace-dsn=https://8FpWBspz0xRNjOGok5vKGA@api.uptrace.dev?grpc=4317"

# Enable gzip compression.
export OTEL_EXPORTER_OTLP_COMPRESSION=gzip

# Enable exponential histograms.
export OTEL_EXPORTER_OTLP_METRICS_DEFAULT_HISTOGRAM_AGGREGATION=BASE2_EXPONENTIAL_BUCKET_HISTOGRAM

# Prefer delta temporality.
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=DELTA


#export OTEL_EXPORTER_OTLP_ENDPOINT=http://cylon-collector1.aws-cylondata.com:4317
#export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_SERVICE_NAME=cylon

exec opentelemetry-instrument --logs_exporter otlp python /cylon/target/aws/scripts/ecs/S3_run_script.py