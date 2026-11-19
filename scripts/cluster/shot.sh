set -e
V=/tmp/pw_$LSB_JOBID; export PLAYWRIGHT_BROWSERS_PATH=/tmp/pwb_$LSB_JOBID UV_CACHE_DIR=/tmp/uvc_$LSB_JOBID
uv venv -q -p 3.11 $V && uv pip install -q -p $V/bin/python playwright
$V/bin/python -m playwright install chromium >/dev/null 2>&1 || $V/bin/python -m playwright install chromium-headless-shell
$V/bin/python scripts/shot.py "$@" || st=$?
rm -rf $V $PLAYWRIGHT_BROWSERS_PATH $UV_CACHE_DIR; exit ${st:-0}
