"""Gymnasium RL environment built on the SAME HastaCore (same dynamics,
same action bound, same hidden scorer for reporting).

Tracks
  obs_mode="state"  privileged: joint positions + current/target hand-frame
                    keypoints + current/target orientation
  obs_mode="image"  image-only: current renders (front, three_quarter) and the
                    target reference images, 64x64 RGB each -> uint8 (12,64,64)

Targets
  target_set="procedural"  random hasta-like poses (training distribution; the
                           28 classical hastas are never seen)
  target_set="hastas"      the benchmark targets (evaluation / per-task training)

Dense reward = decrease of a potential  phi = -(kp_err/10mm) - orient/20deg
- contact gap/5mm - penetration/3mm, plus +5 on success at episode end.
Benchmark *reporting* uses the hidden final-state scorer only.
"""
from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from PIL import Image

from .core import HastaCore, load_targets
from .model import finger_joint_names, DEG, HandParams, KEYPOINTS
from .scorer import score_sim, rot_angle_deg

RES = 64
VIEWS = ("front", "three_quarter")
PRIMS = {"E": (0, 0, 0), "S": (25, 12, 8), "C": (30, 40, 30), "B": (75, 85, 45), "F": (88, 100, 60), "H": (10, 95, 70)}


def euler_rot(core, roll, pitch, yaw):
    s = core.sim
    q = s.q()
    for k, v in zip(["roll", "pitch", "yaw"], [roll, pitch, yaw]):
        q[s.joint_names.index("root_" + k)] = v
    s.set_kinematic(q)
    return s.hand_rot()


def procedural_target(core: HastaCore, rng, with_images=False):
    """Hasta-like random pose: each finger picks a primitive (+noise), thumb random,
    orientation within +-60 deg roll / +-30 pitch,yaw. Settled physically."""
    s = core.sim
    q = np.zeros(s.n)
    for f in ["index", "middle", "ring", "little"]:
        p = PRIMS[rng.choice(list(PRIMS))]
        for v, j in zip(p, ["mcp", "pip", "dip"]):
            q[s.joint_names.index(f"{f}_{j}")] = (v + rng.normal(0, 8)) * DEG
        q[s.joint_names.index(f"{f}_abd")] = rng.uniform(-15, 15) * DEG
    for j, (lo, hi) in {"cmc_flex": (-25, 60), "cmc_abd": (-40, 55), "cmc_rot": (-30, 30), "mcp": (-10, 55), "ip": (-10, 70)}.items():
        q[s.joint_names.index("thumb_" + j)] = rng.uniform(lo, hi) * DEG
    for j, r in {"roll": 60, "pitch": 30, "yaw": 30}.items():
        q[s.joint_names.index("root_" + j)] = rng.uniform(-r, r) * DEG
    q = np.clip(q, s.lo, s.hi)
    s.reset(q)
    s.command(q, steps=300)
    t = dict(id="proc", q=s.q().tolist(), keypoints_hand=s.keypoints().tolist(), hand_rot=s.hand_rot().tolist(),
             contacts=[], apart=[])
    if with_images:
        t["_img"] = [s.render(v, size=RES) for v in VIEWS]
    return t


class HastaGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, obs_mode="state", target_set="procedural", horizon=10, hasta_ids=None,
                 hand_params=None, randomize_hand=False, seed_offset=0, action_mode="absolute"):
        super().__init__()
        self.core = HastaCore(params={"right": hand_params} if hand_params else None, img=RES)
        self.obs_mode, self.target_set, self.horizon = obs_mode, target_set, horizon
        self.randomize_hand = randomize_hand
        # "absolute": a in [-1,1] names a servo target inside the joint range (the
        # semantics of the VLM set_joint tool); "delta": a is the raw step_vector.
        # Either way the env calls the same core.step_vector -> identical dynamics.
        self.action_mode = action_mode
        s = self.core.sim
        self.ctrl_names = finger_joint_names() + ["root_roll", "root_pitch", "root_yaw"]
        self.ctrl_idx = np.array([s.joint_names.index(n) for n in self.ctrl_names])
        self.action_space = gym.spaces.Box(-1, 1, (len(self.ctrl_idx),), np.float32)
        if obs_mode == "state":
            dim = len(self.ctrl_idx) + 2 * 3 * len(KEYPOINTS) + 2 * 9 + 1
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        else:
            self.observation_space = gym.spaces.Box(0, 255, (12, RES, RES), np.uint8)
        self.T = load_targets() if target_set == "hastas" else None
        self.hasta_ids = hasta_ids or (list(self.T) if self.T else None)
        self.rng = np.random.default_rng(seed_offset)
        self.t = 0
        self.target = None

    # ------------------------------------------------------------------
    def _target_images(self, t):
        if "_img" in t:
            return t["_img"]
        return [np.array(Image.open(os.path.join(t["_dir"], t["images"][v])).convert("RGB").resize((RES, RES))) for v in VIEWS]

    def _obs(self):
        s = self.core.sim
        if self.obs_mode == "state":
            q = s.q()[self.ctrl_idx]
            qn = 2 * (q - s.lo[self.ctrl_idx]) / self.core.width[self.ctrl_idx] - 1
            kp = s.keypoints().ravel() * 10
            return np.concatenate([qn, kp, self.kp_t.ravel() * 10, s.hand_rot().ravel(), self.R_t.ravel(),
                                   [1 - self.t / self.horizon]]).astype(np.float32)
        cur = [s.render(v, size=RES) for v in VIEWS]
        return np.concatenate([np.transpose(x, (2, 0, 1)) for x in cur] + list(self.timgs), 0).astype(np.uint8)

    def _phi(self):
        s = self.core.sim
        kp = np.linalg.norm(s.keypoints() - self.kp_t, axis=1)[1:].mean() * 1000
        ori = rot_angle_deg(s.hand_rot(), self.R_t)
        gap = sum(max(0, s.geom_dist(a, b) - 0.003) for a, b in self.target["contacts"]) * 1000
        pen = s.penetration() * 1000
        return -(kp / 10 + ori / 20 + gap / 5 + max(0, pen - 1) / 3)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        if self.randomize_hand:
            hp = HandParams.sample(self.rng)
            self.core = HastaCore(params={"right": hp}, img=RES)
        if self.target_set == "procedural":
            self.target = procedural_target(self.core, self.rng, with_images=self.obs_mode == "image")
        else:
            tid = options.get("target_id") or self.hasta_ids[self.rng.integers(len(self.hasta_ids))]
            self.target = self.T[tid]
        ep_seed = options.get("init_seed", int(self.rng.integers(1 << 30)))
        self.core.reset(self.target, ep_seed)
        self.kp_t = np.array(self.target["keypoints_hand"])
        self.R_t = np.array(self.target["hand_rot"])
        if self.obs_mode == "image":
            self.timgs = [np.transpose(x, (2, 0, 1)) for x in self._target_images(self.target)]
        self.t = 0
        self.phi = self._phi()
        return self._obs(), {}

    def step(self, action):
        a = np.zeros(self.core.n)
        action = np.clip(np.asarray(action, float), -1, 1)
        if self.action_mode == "absolute":
            s = self.core.sim
            i = self.ctrl_idx
            want = s.lo[i] + (action + 1) / 2 * (s.hi[i] - s.lo[i])
            a[i] = (want - s.target()[i]) / (self.core.STEP_SCALE * self.core.width[i])
        else:
            a[self.ctrl_idx] = action
        self.core.step_vector(a)
        self.t += 1
        phi = self._phi()
        r = phi - self.phi
        self.phi = phi
        trunc = self.t >= self.horizon
        info = {}
        if trunc:
            sc = score_sim(self.core.sim, self.target)
            info["score"] = sc
            r += 5.0 * sc["success"]
        return self._obs(), float(r), False, trunc, info
