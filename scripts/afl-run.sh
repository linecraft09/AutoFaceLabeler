#!/bin/bash
# AFL Pipeline wrapper — skip if already completed
MARKER="/data/afl_logs/.pipeline-completed"

if [ -f "$MARKER" ]; then
    echo "$(date -Is) Pipeline already completed. Remove $MARKER to re-run."
    exit 0
fi

cd /root/codex_workspace/AutoFaceLabeler
/root/miniconda3/envs/afl/bin/python src/main.py
exit_code=$?

if [ $exit_code -eq 0 ]; then
    touch "$MARKER"
    echo "$(date -Is) Pipeline completed. Marker set at $MARKER"
fi

exit $exit_code
