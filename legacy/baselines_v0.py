"""Non-LLM baselines on the benchmark track.

  noop          submit the random start pose
  random_image  image-only random local search (accept if silhouette IoU vs. the
                target images improves); budget = number of simulator actions
  es_image      image-only (1+lambda)-ES with step-size adaptation, same loss
  ik_oracle     PRIVILEGED: least squares on the hidden 3-D target keypoints +
                orientation (upper bound; not a fair competitor)

python -m hastabench.baselines --method random_image --budget 40 --targets all28 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from hastabench.core import HastaCore, load_targets
from hastabench.hastas import CORE8, ALL28
from hastabench.model import finger_joint_names, DEG
from hastabench.scorer import rot_angle_deg

VIEWS = ("front", "three_quarter")
RES = 128


def skin_mask(img):
    img = img.astype(np.int16)
    return (img[..., 0] - img[..., 2]) > 45


class ImageLoss:
    def __init__(self, core: HastaCore, target):
        self.core = core
        self.views = tuple(target.get("views", VIEWS))
        self.tm = [skin_mask(np.array(Image.open(os.path.join(target["_dir"], target["images"][v])).convert("RGB").resize((RES, RES))))
                   for v in self.views]

    def __call__(self):
        l = 0.0
        for v, tm in zip(self.views, self.tm):
            m = skin_mask(self.core.sim.render(v, size=RES))
            inter, uni = (m & tm).sum(), (m | tm).sum()
            l += 1.0 - inter / max(uni, 1)
        return l / len(self.views)


def opt_index(core):
    s = core.sim
    names = []
    for p in s.prefixes:
        names += finger_joint_names(p) + [p + "root_roll", p + "root_pitch", p + "root_yaw"]
        if len(s.prefixes) == 2:
            names += [p + "root_tx", p + "root_ty", p + "root_tz"]
    return np.array([s.joint_names.index(n) for n in names])


def run_search(core, target, seed, budget, method):
    rng = np.random.default_rng(1000 + seed)
    core.reset(target, seed)
    loss = ImageLoss(core, target)
    idx = opt_index(core)
    w = core.width[idx]
    best_t = core.sim.target()
    best_state = core.sim.q()
    best = loss()
    sigma = 0.15 if method == "random_image" else 0.25
    used = 0
    succ_hist = []
    while used < budget:
        a = np.zeros(core.n)
        if method == "random_image":
            k = rng.integers(1, 5)                     # perturb a few joints
            j = rng.choice(len(idx), k, replace=False)
            a[idx[j]] = rng.normal(0, sigma, k)
        else:                                          # ES: perturb all, 1/5th rule
            a[idx] = rng.normal(0, sigma, len(idx)) * (0.3 + 0.7 * rng.random(len(idx)))
        core.sim.reset(best_state); core.sim.d.ctrl[core.sim.act_ids] = best_t
        core.step_vector(a)
        used += 1
        l = loss()
        if l < best:
            best, best_t, best_state = l, core.sim.target(), core.sim.q()
            succ_hist.append(1)
        else:
            succ_hist.append(0)
        if method == "es_image" and len(succ_hist) >= 10:
            rate = np.mean(succ_hist[-10:])
            sigma = float(np.clip(sigma * (1.15 if rate > 0.2 else 0.87), 0.01, 0.5))
            succ_hist = succ_hist[-10:] if len(succ_hist) > 50 else succ_hist
    core.sim.reset(best_state); core.sim.d.ctrl[core.sim.act_ids] = best_t
    core.sim.command(best_t)
    return core.evaluate(), dict(image_loss=float(best), actions=used)


def run_ik(core, target, seed):
    core.reset(target, seed)
    s = core.sim
    idx = opt_index(core)
    kp_t = np.array(target["keypoints_hand"])
    R_t = np.array(target["hand_rot"])
    q0 = s.q()

    def res(x):
        q = q0.copy(); q[idx] = x
        s.set_kinematic(q)
        r1 = (s.keypoints() - kp_t).ravel() * 100
        r2 = (s.hand_rot() - R_t).ravel() * 2
        cd = [30 * max(0, s.geom_dist(a, b) - 0.001) for a, b in target["contacts"]]
        return np.concatenate([r1, r2, cd])
    sol = least_squares(res, q0[idx], bounds=(s.lo[idx], s.hi[idx]), diff_step=1e-3, max_nfev=300)
    q = q0.copy(); q[idx] = sol.x
    s.reset(q0)
    s.command(q0)
    core.step_vector((q - s.target()) / core.width)   # one action
    return core.evaluate(), dict(actions=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["noop", "random_image", "es_image", "ik_oracle"])
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--targets", default="all28")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    if args.targets == "samyuta":
        T = load_targets("samyuta_v1")
        names = [k for k, v in T.items() if v["valid"]]
        core = HastaCore(hands=("right", "left"))
    else:
        T = load_targets()
        names = CORE8 if args.targets == "core8" else ALL28
        core = HastaCore()
    os.makedirs("results/baselines", exist_ok=True)
    out = f"results/baselines/{args.method}_b{args.budget}_{args.targets}.jsonl"
    with open(out, "w") as f:
        for s in args.seeds:
            for n in names:
                if args.method == "noop":
                    core.reset(T[n], s); sc, info = core.evaluate(), {}
                elif args.method == "ik_oracle":
                    sc, info = run_ik(core, T[n], s)
                else:
                    sc, info = run_search(core, T[n], s, args.budget, args.method)
                f.write(json.dumps(dict(method=args.method, budget=args.budget, target=n, seed=s, score=sc, **info)) + "\n")
                print(n, s, sc["success"], round(sc["hasta_score"], 3), round(sc["kp_err_mm"], 1), round(sc["orient_err_deg"], 1), flush=True)


if __name__ == "__main__":
    main()
