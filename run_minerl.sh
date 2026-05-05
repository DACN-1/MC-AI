#!/usr/bin/env bash
set -e

DISPLAY_NUM=:1
Xvfb $DISPLAY_NUM -screen 0 1024x768x24 &
XVFB_PID=$!
export DISPLAY=$DISPLAY_NUM

echo "[run_minerl.sh] Xvfb started (PID $XVFB_PID, display $DISPLAY)"
python3.10 "$@"
EXIT_CODE=$?

kill $XVFB_PID 2>/dev/null
exit $EXIT_CODE
