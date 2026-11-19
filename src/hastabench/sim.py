"""HandSim: MuJoCo wrapper shared by the benchmark and the RL environment."""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from .model import build_mjcf, finger_joint_names, root_joint_names, KEYPOINTS, HandParams, DEG

SETTLE_STEPS = 150


class HandSim:
    def __init__(self, hands=("right",), params: dict | None = None, img=384):
        self.hands = tuple(hands)
        self.two = len(self.hands) == 2
        self.prefixes = [("R_" if h == "right" else "L_") if self.two else "" for h in self.hands]
        self.xml = build_mjcf(self.hands, params, img, img)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.img = img
        self._renderers = {}
        # ordered joint list = action/qpos layout used everywhere
        self.joint_names = []
        for p in self.prefixes:
            self.joint_names += root_joint_names(p) + finger_joint_names(p)
        self.jids = np.array([self.m.joint(n).id for n in self.joint_names])
        self.qadr = self.m.jnt_qposadr[self.jids]
        self.dadr = self.m.jnt_dofadr[self.jids]
        self.lo = self.m.jnt_range[self.jids, 0].copy()
        self.hi = self.m.jnt_range[self.jids, 1].copy()
        self.is_angle = np.array([self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in self.jids])
        self.act_ids = np.array([self.m.actuator(f"a_{n}").id for n in self.joint_names])
        self.kp_sites = {p: np.array([self.m.site(f"{p}kp_{k}").id for k in KEYPOINTS]) for p in self.prefixes}
        self.palm_body = {p: self.m.body(f"{p}root").id for p in self.prefixes}
        self.geom_names = [self.m.geom(i).name for i in range(self.m.ngeom)]
        self.reset()

    # ------------------------------------------------------------ state
    @property
    def n(self):
        return len(self.joint_names)

    def reset(self, q=None):
        mujoco.mj_resetData(self.m, self.d)
        q = np.zeros(self.n) if q is None else np.asarray(q, float)
        q = np.clip(q, self.lo, self.hi)
        self.d.qpos[self.qadr] = q
        self.d.ctrl[self.act_ids] = q
        mujoco.mj_forward(self.m, self.d)

    def q(self):
        return self.d.qpos[self.qadr].copy()

    def target(self):
        return self.d.ctrl[self.act_ids].copy()

    def set_kinematic(self, q):
        """Teleport (no dynamics). Used by target construction / IK only."""
        q = np.clip(np.asarray(q, float), self.lo, self.hi)
        self.d.qpos[self.qadr] = q
        self.d.qvel[:] = 0
        self.d.ctrl[self.act_ids] = q
        mujoco.mj_forward(self.m, self.d)

    def command(self, q_target, steps=SETTLE_STEPS):
        """Set servo targets and integrate until the hand settles.

        Collisions are resolved physically: a finger pushed into another
        finger stops at the contact instead of passing through it."""
        q_target = np.clip(np.asarray(q_target, float), self.lo, self.hi)
        self.d.ctrl[self.act_ids] = q_target
        for _ in range(steps):
            mujoco.mj_step(self.m, self.d)
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        return self.q()

    # ------------------------------------------------------------ queries
    def keypoints(self, prefix=None, frame="hand"):
        """21x3 keypoints. frame='hand' -> expressed in that hand's root frame
        (translation/rotation invariant); 'world' -> world coordinates."""
        prefix = self.prefixes[0] if prefix is None else prefix
        P = self.d.site_xpos[self.kp_sites[prefix]].copy()
        if frame == "world":
            return P
        b = self.palm_body[prefix]
        R = self.d.xmat[b].reshape(3, 3)
        return (P - self.d.xpos[b]) @ R

    def hand_rot(self, prefix=None):
        prefix = self.prefixes[0] if prefix is None else prefix
        return self.d.xmat[self.palm_body[prefix]].reshape(3, 3).copy()

    def hand_pos(self, prefix=None):
        prefix = self.prefixes[0] if prefix is None else prefix
        return self.d.xpos[self.palm_body[prefix]].copy()

    def geom_dist(self, g1: str, g2: str, distmax=0.2):
        fromto = np.zeros(6)
        return mujoco.mj_geomDistance(self.m, self.d, self.m.geom(g1).id, self.m.geom(g2).id, distmax, fromto)

    def penetration(self):
        """Max penetration depth (m, >=0) over active contacts."""
        if self.d.ncon == 0:
            return 0.0
        return float(max(0.0, -min(c.dist for c in self.d.contact[: self.d.ncon])))

    def contacts(self, thresh=0.0):
        out = []
        for c in self.d.contact[: self.d.ncon]:
            if c.dist <= thresh:
                out.append((self.geom_names[c.geom1], self.geom_names[c.geom2], float(c.dist)))
        return out

    # ------------------------------------------------------------ render
    def render(self, camera="front", size=None, keypoints=False):
        size = size or self.img
        key = size
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.m, size, size)
        r = self._renderers[key]
        opt = mujoco.MjvOption()
        if keypoints:
            opt.sitegroup[4] = 1
        r.update_scene(self.d, camera=camera, scene_option=opt)
        return r.render().copy()

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers = {}

    # ------------------------------------------------------------ helpers
    def to_internal(self, name, value):
        """Human units (deg / metres) -> qpos units (rad / metres)."""
        i = self.joint_names.index(name)
        return value * DEG if self.is_angle[i] else value

    def to_human(self, q):
        q = np.asarray(q, float)
        return np.where(self.is_angle, q / DEG, q)

    def from_human(self, qh):
        qh = np.asarray(qh, float)
        return np.where(self.is_angle, qh * DEG, qh)
