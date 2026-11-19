#!/bin/bash
# usage: bsub ... scripts/cluster/job.sh <python args...>   (runs with home venv, EGL on GPU)
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl OMP_NUM_THREADS=4 PYTHONPATH=src PYTHONUNBUFFERED=1
exec .venv/bin/python "$@"
