"""Scorer reliability: (1) confusion — does hasta i's pose 'succeed' on target j?
(2) robustness — success under small joint noise on the true target."""
import json, sys
import numpy as np
from hastabench.core import HastaCore, load_targets
from hastabench.model import finger_joint_names, DEG
from hastabench.scorer import score_sim

T = load_targets()
names = list(T)
core = HastaCore()
sim = core.sim
fidx = [sim.joint_names.index(j) for j in finger_joint_names()]
ridx = [i for i in range(sim.n) if i not in fidx]
N = len(names)
succ = np.zeros((N, N), int); kp = np.zeros((N, N))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        q = np.array(T[b]["q"]); q[fidx] = np.array(T[a]["q"])[fidx]   # shape of a, orientation of b
        sim.set_kinematic(q); sim.command(q)
        r = score_sim(sim, T[b]); succ[i, j] = r["success"]; kp[i, j] = r["kp_err_mm"]
rob = {}
rng = np.random.default_rng(0)
for sig in [2, 4, 6, 10]:
    ok = []
    for b in names:
        for k in range(5):
            q = np.array(T[b]["q_command"]); q[fidx] += rng.normal(0, sig * DEG, len(fidx))
            q[ridx[3:]] += rng.normal(0, sig * DEG, 3)
            sim.set_kinematic(np.array(T[b]["q"])); sim.command(q)
            ok.append(score_sim(sim, T[b])["success"])
    rob[sig] = float(np.mean(ok))
off = succ[~np.eye(N, dtype=bool)]
mink = np.where(np.eye(N, dtype=bool), np.inf, kp).min(1)
out = dict(diag_success=int(np.trace(succ)), n=N, offdiag_false_success=int(off.sum()),
           confusable_pairs=[(names[i], names[j]) for i in range(N) for j in range(N) if i != j and succ[i, j]],
           nearest_other_kp_mm={n: round(float(v), 1) for n, v in zip(names, mink)},
           noise_success={f"{k}deg": v for k, v in rob.items()})
print(json.dumps(out, indent=1))
json.dump(dict(out, kp_matrix=kp.round(1).tolist(), names=names), open("results/scorer_validation.json", "w"), indent=1)
