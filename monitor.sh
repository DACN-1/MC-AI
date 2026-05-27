#!/bin/bash
# Quick at-a-glance training status. Usage:
#   ./monitor.sh           # one-shot snapshot
#   ./monitor.sh watch     # refresh every 10 s
#   ./monitor.sh tail      # tail the currently running job's stdout

set -euo pipefail

cd "$(dirname "$0")"
ME=${USER:-$(whoami)}

snapshot() {
    echo "=== $(date) ==="
    echo
    echo "-- squeue --"
    squeue -u "$ME" -o "%.10i %.30j %.2t %.10M %.10L %R" || echo "(no jobs)"
    echo
    echo "-- Abaki nodes --"
    sinfo -p Abaki -o "%n %t %C"
    echo
    echo "-- completed cells (output/<cell>/metrics.json) --"
    if compgen -G "output/*/metrics.json" > /dev/null; then
        for f in output/*/metrics.json; do
            cell=$(basename "$(dirname "$f")")
            ts=$(date -r "$f" "+%Y-%m-%d %H:%M")
            bin_acc=$(python3 -c "import json;print(round(json.load(open('$f'))['binary_accuracy'],4))" 2>/dev/null || echo "?")
            cam_mae=$(python3 -c "import json;print(round(json.load(open('$f'))['camera_mae_degrees'],3))" 2>/dev/null || echo "?")
            printf "  %-36s  bin_acc=%-6s  cam_mae=%-6s  %s\n" "$cell" "$bin_acc" "$cam_mae" "$ts"
        done
    else
        echo "  (none yet)"
    fi
    echo
    echo "-- last 10 lines of newest running job stdout --"
    running=$(squeue -u "$ME" -h -t RUNNING -o "%i" | head -1 || true)
    if [ -n "$running" ] && [ -f "logs/slurm_${running}.out" ]; then
        echo "  job $running:"
        tail -10 "logs/slurm_${running}.out" | sed 's/^/    /'
    else
        echo "  (no running job)"
    fi
}

watch_loop() {
    while true; do
        clear
        snapshot
        sleep 10
    done
}

tail_running() {
    running=$(squeue -u "$ME" -h -t RUNNING -o "%i" | head -1 || true)
    if [ -z "$running" ]; then
        echo "No running job."
        exit 1
    fi
    echo "Tailing logs/slurm_${running}.out (Ctrl-C to stop)…"
    tail -F "logs/slurm_${running}.out"
}

case "${1:-snapshot}" in
    snapshot) snapshot ;;
    watch)    watch_loop ;;
    tail)     tail_running ;;
    *)        echo "Usage: $0 [snapshot|watch|tail]" >&2; exit 1 ;;
esac
