#!/bin/bash
# RL job: torch venv lives on node-local /tmp (home quota is tight)
set -e
cd "$(dirname "$0")/../.."
V=/tmp/hb_venv_$LSB_JOBID
export UV_CACHE_DIR=/tmp/uvcache_$LSB_JOBID
uv venv -q -p 3.11 $V
uv pip install -q -p $V/bin/python -e . torch stable-baselines3 2>&1 | tail -2
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=src PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
$V/bin/python "$@" || st=$?
rm -rf $V $UV_CACHE_DIR
exit ${st:-0}
