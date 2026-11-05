#!/usr/bin/env bash
# ==============================================================================
# Neev: Lightning AI One-Click Production Launcher
# Sets up dependencies, launches Uvicorn server, and starts keep-alive daemon
# ==============================================================================

set -e

echo "=== [1/3] Installing Dependencies ==="
pip install -r requirements.txt

echo "=== [2/3] Starting Neev Uvicorn Server in Background ==="
nohup python -m uvicorn neev.app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

echo "=== [3/3] Starting Keep-Alive Loop (Prevents Auto-Sleep) ==="
nohup bash -c 'while true; do curl -s http://localhost:8000/health > /dev/null || true; sleep 300; done' > /dev/null 2>&1 &
KEEPALIVE_PID=$!
echo "Keep-alive daemon started with PID: $KEEPALIVE_PID"

echo ""
echo "=========================================================="
echo "  NEEV IS RUNNING ON PORT 8000 (16 GB RAM)"
echo "  1. Open the 'Port viewer' plugin in the right panel"
echo "  2. Enter Port: 8000 and switch to 'Public'"
echo "  3. Open your public HTTPS link"
echo "=========================================================="
