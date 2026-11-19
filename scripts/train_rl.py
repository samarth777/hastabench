"""PPO on HastaGymEnv. Trains on PROCEDURAL targets (classical hastas held out)
or directly on the hastas, then evaluates on the 28 benchmark hastas with the
benchmark's start configurations (init seeds 0,1,2) and on held-out hand
proportions.

python scripts/train_rl.py --obs state --train procedural --steps 3000000 --out results/rl/state_proc
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from hastabench.gym_env import HastaGymEnv
from hastabench.core import HastaCore, load_targets
from hastabench.model import HandParams
from hastabench.hastas import ALL28, CORE8


def make(obs, train, horizon, rank, randomize_hand):
    def f():
        return HastaGymEnv(obs_mode=obs, target_set=train, horizon=horizon, seed_offset=10_000 + rank,
                           randomize_hand=randomize_hand)
    return f


def episode(model, env, tid, init_seed):
    o, _ = env.reset(options=dict(target_id=tid, init_seed=init_seed))
    done = False
    while not done:
        a, _ = model.predict(o, deterministic=True)
        o, r, term, trunc, info = env.step(a)
        done = term or trunc
    return info["score"]


def evaluate(model, obs, horizon, seeds=(0, 1, 2), hand_params=None, targets=None):
    env = HastaGymEnv(obs_mode=obs, target_set="hastas", horizon=horizon, hand_params=hand_params)
    if targets is not None:      # per-hand regenerated targets (proportion test)
        env.T = targets
        env.hasta_ids = list(targets)
    rows = []
    for s in seeds:
        for t in ALL28:
            sc = episode(model, env, t, s)
            rows.append(dict(target=t, seed=s, success=sc["success"], hasta_score=sc["hasta_score"],
                             kp_err_mm=sc["kp_err_mm"], orient_err_deg=sc["orient_err_deg"],
                             constraint_frac=sc["constraint_frac"], valid=sc["valid"]))
    return rows


def variant_targets(hp, base):
    """Re-pose each hasta's joint configuration on a different hand; the target
    keypoints/images come from that hand (shape = joint configuration)."""
    from PIL import Image
    import tempfile
    core = HastaCore(params={"right": hp}, img=384)
    d = tempfile.mkdtemp(prefix="hb_var_")
    out = {}
    for k, t in base.items():
        q = np.array(t["q_command"])
        core.sim.set_kinematic(q)
        core.sim.command(q, steps=400)
        r = dict(t)
        r.update(keypoints_hand=core.sim.keypoints().tolist(), hand_rot=core.sim.hand_rot().tolist(), _dir=d,
                 images={v: f"{k}_{v}.png" for v in ("front", "three_quarter")})
        for v in ("front", "three_quarter"):
            Image.fromarray(core.sim.render(v)).save(os.path.join(d, f"{k}_{v}.png"))
        out[k] = r
    return out


def summarize(rows):
    a = lambda k: float(np.mean([r[k] for r in rows]))
    core = [r for r in rows if r["target"] in CORE8]
    return dict(n=len(rows), success=a("success"), hasta_score=a("hasta_score"), kp_err_mm=a("kp_err_mm"),
                orient_err_deg=a("orient_err_deg"), core8_success=float(np.mean([r["success"] for r in core])))


class EvalCB(BaseCallback):
    def __init__(self, obs, horizon, every, out):
        super().__init__()
        self.obs, self.horizon, self.every, self.out, self.next = obs, horizon, every, out, every
        self.hist = []

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            s = summarize(evaluate(self.model, self.obs, self.horizon, seeds=(0,)))
            s["steps"] = self.num_timesteps
            self.hist.append(s)
            print("EVAL", json.dumps(s), flush=True)
            json.dump(self.hist, open(os.path.join(self.out, "curve.json"), "w"), indent=1)
            self.model.save(os.path.join(self.out, "model"))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="state")
    ap.add_argument("--train", default="procedural")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--randomize-hand", action="store_true")
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    venv = VecMonitor(SubprocVecEnv([make(args.obs, args.train, args.horizon, i, args.randomize_hand) for i in range(args.n_envs)]))
    if args.obs == "state":
        model = PPO("MlpPolicy", venv, n_steps=128, batch_size=4096, n_epochs=8, learning_rate=3e-4, gamma=0.95,
                    gae_lambda=0.95, ent_coef=0.0, clip_range=0.2, policy_kwargs=dict(net_arch=[512, 512, 256], log_std_init=-1.0),
                    verbose=1, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = PPO("CnnPolicy", venv, n_steps=128, batch_size=1024, n_epochs=4, learning_rate=2.5e-4, gamma=0.95,
                    gae_lambda=0.95, ent_coef=0.0, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0), verbose=1, device="cuda")
    t0 = time.time()
    model.learn(args.steps, callback=EvalCB(args.obs, args.horizon, args.eval_every, args.out))
    model.save(os.path.join(args.out, "model"))
    train_min = (time.time() - t0) / 60
    res = dict(args=vars(args), train_minutes=train_min)
    rows = evaluate(model, args.obs, args.horizon)
    res["nominal"] = summarize(rows)
    json.dump(rows, open(os.path.join(args.out, "eval_nominal.json"), "w"))
    # held-out hand proportions (5 sampled hands, seed 0 init)
    base = load_targets()
    prop = []
    for k in range(5):
        hp = HandParams.sample(np.random.default_rng(500 + k))
        rows_k = evaluate(model, args.obs, args.horizon, seeds=(0,), hand_params=hp, targets=variant_targets(hp, base))
        prop.append(summarize(rows_k))
    res["heldout_proportions"] = prop
    res["heldout_proportions_mean_success"] = float(np.mean([p["success"] for p in prop]))
    print("FINAL", json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(args.out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
