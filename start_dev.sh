#!/bin/bash
# Development startup script for MGT Eval Web Frontend

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# Load env from project root if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Runtime path defaults (all can be overridden in .env)
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export FRONTEND_PORT="${FRONTEND_PORT:-3000}"
export MGT_EVAL_BACKEND_TMP_DIR="${MGT_EVAL_BACKEND_TMP_DIR:-$HOME/.mgt_eval/backend_tmp}"
export MGT_EVAL_DEMO_TMP_ROOT="${MGT_EVAL_DEMO_TMP_ROOT:-$HOME/.mgt_eval/demo_tmp}"
export MGT_EVAL_RUNTIME_CACHE_ROOT="${MGT_EVAL_RUNTIME_CACHE_ROOT:-$HOME/.cache/mgt_eval_cache}"
export MGT_EVAL_HF_CACHE_READ_DIR="${MGT_EVAL_HF_CACHE_READ_DIR:-$HOME/.cache/huggingface/hub}"
export PUBLIC_AUTH_DB_PATH="${PUBLIC_AUTH_DB_PATH:-$HOME/.uncoverai/uncoverai_public.db}"

mkdir -p "$MGT_EVAL_BACKEND_TMP_DIR" "$MGT_EVAL_DEMO_TMP_ROOT" "$MGT_EVAL_RUNTIME_CACHE_ROOT"
mkdir -p "$(dirname "$PUBLIC_AUTH_DB_PATH")"

echo "=========================================="
echo "MGT Eval Web Frontend - Development Mode"
echo "=========================================="
echo "Working directory: $SCRIPT_DIR"
echo "Backend port: $BACKEND_PORT"
echo "Frontend port: $FRONTEND_PORT"
echo "MGT_EVAL_BACKEND_TMP_DIR: $MGT_EVAL_BACKEND_TMP_DIR"
echo "MGT_EVAL_DEMO_TMP_ROOT: $MGT_EVAL_DEMO_TMP_ROOT"
echo "MGT_EVAL_RUNTIME_CACHE_ROOT: $MGT_EVAL_RUNTIME_CACHE_ROOT"
echo "MGT_EVAL_HF_CACHE_READ_DIR: $MGT_EVAL_HF_CACHE_READ_DIR"
echo "PUBLIC_AUTH_DB_PATH: $PUBLIC_AUTH_DB_PATH"
echo ""

# Check if backend dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing backend dependencies..."
    pip install -r backend/requirements.txt
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

echo ""
echo "Starting backend on http://localhost:$BACKEND_PORT..."
echo "Starting frontend on http://localhost:$FRONTEND_PORT..."
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    local exit_code=$?
    trap - INT TERM EXIT

    # Kill whole process groups first (uvicorn reloader / vite child processes).
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill -TERM -- "-$FRONTEND_PID" 2>/dev/null || true
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [ -n "${BACKEND_PID:-}" ]; then
        kill -TERM -- "-$BACKEND_PID" 2>/dev/null || true
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
    fi

    wait 2>/dev/null || true
    exit "$exit_code"
}

trap cleanup INT TERM EXIT

# Start backend in background
setsid python -m backend.main &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Wait for user interrupt
wait -n "$BACKEND_PID" "$FRONTEND_PID"
