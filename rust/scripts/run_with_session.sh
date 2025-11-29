#!/bin/bash
# Helper script to run Cylon programs with proper session management
#
# This prevents segfaults from stale UCX worker addresses in Redis by ensuring
# each run uses a unique session ID.
#
# Usage:
#   ./run_with_session.sh mpirun -n 4 ./my_program
#   ./run_with_session.sh cargo test --features ucx -- --ignored

# Generate unique session ID
if command -v uuidgen &> /dev/null; then
    export CYLON_SESSION_ID=$(uuidgen)
else
    # Fallback if uuidgen not available
    export CYLON_SESSION_ID="session_$(date +%s)_$$"
fi

# Set default Redis URL if not already set
export CYLON_REDIS_URL="${CYLON_REDIS_URL:-redis://localhost:6379}"

echo "======================================"
echo "Cylon Session Manager"
echo "======================================"
echo "Session ID: $CYLON_SESSION_ID"
echo "Redis URL:  $CYLON_REDIS_URL"
echo "Command:    $@"
echo "======================================"
echo ""

# Run the command with the session environment
exec "$@"
