#!/bin/bash
# Development startup script for MGT Eval Web Frontend

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "=========================================="
echo "MGT Eval Web Frontend - Development Mode"
echo "=========================================="
echo "Working directory: $SCRIPT_DIR"
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
echo "Starting backend on http://localhost:8000..."
echo "Starting frontend on http://localhost:3000..."
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start backend in background
python -m backend.main &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend
cd frontend
npm run dev &
FRONTEND_PID=$!

# Wait for user interrupt
wait

# Cleanup
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
