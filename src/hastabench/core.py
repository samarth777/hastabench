"""HastaBench core environment.

ONE action representation underlies everything: a normalised vector
a in [-1, 1]^n.  Each step moves the servo targets by a * step_scale *
(joint range), clips to limits, then the physics settles (collisions are
resolved, contacts form).  The VLM tools (set_joint / rotate_wrist /
move_hand) are compiled into exactly such vectors, and the Gymnasium env
feeds policy outputs into the same `step_vector`.  So benchmark and RL
share dynamics, action bounds and the hidden scorer.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .sim import HandSim
from .model import finger_joint_names, root_joint_names, DEG, HandParams
from .scorer import score_sim, score_two

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_targets(version="targets_v1", data_dir=None):
    d = os.path.join(data_dir or DATA, version)
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(p) == "index.json":
            continue
        with open(p) as f:
            r = json.load(f)
        r["_dir"] = d
        out[r["id"]] = r
    return out


class HastaCore:
    """Shared engine. `hands` = ("right",) for asamyuta, ("right","left") for samyuta."""

    STEP_SCALE = 1.0       # one action may traverse a joint's full range

    def __init__(self, hands=("right",), params=None, img=384, budget=40):
        self.sim = HandSim(hands, params, img)
        self.n = self.sim.n
        self.width = self.sim.hi - self.sim.lo
        self.budget = budget
        self.target = None
        self.steps = 0
        self.log = []

    # ---------------------------------------------------------------- reset
    def initial_q(self, seed: int):
        """Seeded start: relaxed open hand(s) with random finger curl and a
        random global orientation (+-35 deg) and offset (+-2 cm)."""
        rng = np.random.default_rng(seed)
        s = self.sim
        q = np.zeros(self.n)
        for p in s.prefixes:
            for j in finger_joint_names(p):
                i = s.joint_names.index(j)
                base = j.split("_")[-1]
                if base in ("mcp", "pip", "dip", "ip"):
                    q[i] = rng.uniform(0, 25) * DEG
                elif base == "abd" and "thumb" not in j:
                    q[i] = rng.uniform(-5, 5) * DEG
                elif "thumb" in j:
                    q[i] = rng.uniform(-10, 15) * DEG
            for k, j in zip(["tx", "ty", "tz", "roll", "pitch", "yaw"], root_joint_names(p)):
                i = s.joint_names.index(j)
                q[i] = rng.uniform(-0.02, 0.02) if k.startswith("t") else rng.uniform(-35, 35) * DEG
        return q

    def reset(self, target: dict, seed: int = 0, q0=None):
        self.target = target
        self.steps = 0
        self.log = []
        q = self.initial_q(seed) if q0 is None else np.asarray(q0)
        self.sim.reset(q)
        self.sim.command(q)
        return self.sim.q()

    # ---------------------------------------------------------------- step
    def step_vector(self, a):
        """Universal action: a in [-1,1]^n -> bounded change of servo targets."""
        a = np.clip(np.asarray(a, float), -1, 1)
        tgt = np.clip(self.sim.target() + a * self.STEP_SCALE * self.width, self.sim.lo, self.sim.hi)
        self.sim.command(tgt)
        self.steps += 1
        return self.sim.q()

    def delta_to_action(self, deltas: dict):
        """{joint_name: delta in internal units} -> normalised action vector."""
        a = np.zeros(self.n)
        for j, d in deltas.items():
            i = self.sim.joint_names.index(j)
            a[i] = d / (self.STEP_SCALE * self.width[i])
        return np.clip(a, -1, 1)

    # ---------------------------------------------------------------- scoring
    def evaluate(self):
        if len(self.sim.prefixes) == 1:
            return score_sim(self.sim, self.target)
        return score_two(self.sim, self.target)

    # ---------------------------------------------------------------- tools
    def _pre(self, hand):
        hand = (hand or "right").lower()
        if len(self.sim.prefixes) == 1:
            if hand not in ("right", "r"):
                raise ValueError("this task has only the right hand")
            return ""
        return "R_" if hand in ("right", "r") else "L_"

    def tool_set_joint(self, hand, joint, angle):
        p = self._pre(hand)
        name = p + joint
        if name not in self.sim.joint_names or "root_" in name:
            raise ValueError(f"unknown finger joint '{joint}'")
        i = self.sim.joint_names.index(name)
        cur = self.sim.target()[i]
        want = float(angle) * DEG
        lo, hi = self.sim.lo[i] / DEG, self.sim.hi[i] / DEG
        self.step_vector(self.delta_to_action({name: want - cur}))
        got = self.sim.q()[i] / DEG
        msg = f"{hand} {joint}: commanded {np.clip(angle, lo, hi):.1f} deg (limits [{lo:.0f}, {hi:.0f}]), settled at {got:.1f} deg"
        if abs(got - np.clip(angle, lo, hi)) > 4:
            msg += " (blocked by contact)"
        return msg

    def tool_rotate_wrist(self, hand, roll=0.0, pitch=0.0, yaw=0.0):
        """Relative rotation (deg) of the whole hand: roll about the forearm
        axis, pitch = flexion/extension, yaw = radial/ulnar tilt."""
        p = self._pre(hand)
        d = {p + "root_roll": roll * DEG, p + "root_pitch": pitch * DEG, p + "root_yaw": yaw * DEG}
        self.step_vector(self.delta_to_action(d))
        q = self.sim.q()
        cur = [q[self.sim.joint_names.index(p + f"root_{k}")] / DEG for k in ("roll", "pitch", "yaw")]
        return f"{hand} hand orientation now roll={cur[0]:.1f}, pitch={cur[1]:.1f}, yaw={cur[2]:.1f} deg"

    def tool_move_hand(self, hand, dx=0.0, dy=0.0, dz=0.0):
        """Relative translation in centimetres (x=right, y=away from front camera, z=up)."""
        p = self._pre(hand)
        d = {p + "root_tx": dx / 100, p + "root_ty": dy / 100, p + "root_tz": dz / 100}
        self.step_vector(self.delta_to_action(d))
        q = self.sim.q()
        cur = [q[self.sim.joint_names.index(p + f"root_{k}")] * 100 for k in ("tx", "ty", "tz")]
        return f"{hand} hand offset now x={cur[0]:.1f}, y={cur[1]:.1f}, z={cur[2]:.1f} cm"

    def tool_render_view(self, camera="front"):
        if camera not in ("front", "side", "top", "three_quarter", "back"):
            raise ValueError("camera must be one of front, side, top, three_quarter, back")
        return self.sim.render(camera)
