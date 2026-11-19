"""Parametric articulated hand -> MJCF.

Hand frame (right hand, rest pose): +z = finger direction, -y = palmar side
(palm normal), +x = radial (thumb) side.  The left hand is an exact mirror
image (x -> -x).  Joint axes are mirrored as pseudo-vectors so the *same*
joint values give mirror-symmetric poses on both hands (positive flexion
always curls toward the palm, positive abduction always spreads radially).

Each hand root has 6 DOF (3 slides + roll/pitch/yaw hinges) driven by
position actuators, like every finger joint.  The simulator is a light
physics model: zero gravity, joint position servos, self-collision on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

FINGERS = ["index", "middle", "ring", "little"]
SEG = ["prox", "mid", "dist"]
DEG = np.pi / 180


@dataclass
class HandParams:
    scale: float = 1.0                    # overall size
    finger_len: dict = field(default_factory=lambda: {"index": 1.0, "middle": 1.0, "ring": 1.0, "little": 1.0, "thumb": 1.0})
    palm_width: float = 1.0               # multiplies palm width / finger spacing
    radius: float = 1.0                   # multiplies phalanx radii
    skin_rgba: tuple = (0.87, 0.68, 0.56, 1.0)

    @staticmethod
    def sample(rng: np.random.Generator, spread: float = 0.12) -> "HandParams":
        """Random hand proportions (for held-out-proportion generalisation)."""
        u = lambda: 1.0 + rng.uniform(-spread, spread)
        return HandParams(scale=1.0 + rng.uniform(-spread, spread) * 0.8,
                          finger_len={k: u() for k in ["index", "middle", "ring", "little", "thumb"]},
                          palm_width=u(), radius=1.0 + rng.uniform(-spread, spread) * 0.6)


# ---------------------------------------------------------------- geometry
# (proximal, middle, distal) lengths in metres for a ~18.5 cm adult hand
PHAL = {"index": (0.040, 0.025, 0.019), "middle": (0.045, 0.028, 0.020),
        "ring": (0.042, 0.027, 0.020), "little": (0.033, 0.020, 0.017)}
THUMB = (0.042, 0.031, 0.025)               # metacarpal, proximal, distal
MCP_X = {"index": 0.0295, "middle": 0.0095, "ring": -0.0100, "little": -0.0285}
MCP_Z = {"index": 0.094, "middle": 0.097, "ring": 0.094, "little": 0.086}
RAD = {"index": 0.0090, "middle": 0.0092, "ring": 0.0087, "little": 0.0078}
THUMB_RAD = 0.0105
PALM_HALF = (0.0405, 0.0120, 0.047)

# joint limits in degrees
LIM = {"abd": (-20, 20), "mcp": (-25, 95), "pip": (0, 110), "dip": (-5, 85)}
THUMB_LIM = {"cmc_flex": (-30, 70), "cmc_abd": (-45, 40), "cmc_rot": (-40, 40), "mcp": (-15, 65), "ip": (-20, 85)}
ROOT_LIM = {"tx": (-0.15, 0.15), "ty": (-0.15, 0.15), "tz": (-0.15, 0.15),
            "roll": (-180, 180), "pitch": (-90, 90), "yaw": (-90, 90)}


def finger_joint_names(prefix: str = "") -> list[str]:
    names = [f"{prefix}thumb_{j}" for j in ["cmc_flex", "cmc_abd", "cmc_rot", "mcp", "ip"]]
    for f in FINGERS:
        names += [f"{prefix}{f}_{j}" for j in ["abd", "mcp", "pip", "dip"]]
    return names


def root_joint_names(prefix: str = "") -> list[str]:
    return [f"{prefix}root_{j}" for j in ["tx", "ty", "tz", "roll", "pitch", "yaw"]]


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def mat2quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(-1))
    return q


class _Emitter:
    """Builds XML for one hand, applying the mirror transform for the left."""

    def __init__(self, side: str, p: HandParams, prefix: str, base_pos):
        self.side, self.p, self.pre = side, p, prefix
        self.M = np.diag([-1.0, 1.0, 1.0]) if side == "left" else np.eye(3)
        self.base_pos = np.asarray(base_pos, float)
        self.lines: list[str] = []
        self.ind = 2

    # mirror helpers ---------------------------------------------------
    def P(self, v):
        return self.M @ np.asarray(v, float)

    def R(self, R):
        return self.M @ R @ self.M

    def A(self, a):
        return -self.M @ np.asarray(a, float) if self.side == "left" else np.asarray(a, float)

    @staticmethod
    def f(v):
        return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))

    def w(self, s):
        self.lines.append("  " * self.ind + s)

    # ------------------------------------------------------------------
    def body_open(self, name, pos, R=None):
        q = "" if R is None else f' quat="{self.f(mat2quat(self.R(R)))}"'
        self.w(f'<body name="{self.pre}{name}" pos="{self.f(self.P(pos))}"{q}>')
        self.ind += 1

    def body_close(self):
        self.ind -= 1
        self.w("</body>")

    def hinge(self, name, axis, lim, cls="finger"):
        self.w(f'<joint name="{self.pre}{name}" class="{cls}" type="hinge" axis="{self.f(self.A(axis))}" '
               f'range="{lim[0]} {lim[1]}"/>')

    def capsule(self, name, length, r, cls="skin"):
        self.w(f'<geom name="{self.pre}{name}" class="{cls}" type="capsule" '
               f'fromto="0 0 0 {self.f(self.P([0, 0, length]))}" size="{r:.5g}"/>')

    def site(self, name, pos, size=0.002):
        self.w(f'<site name="{self.pre}{name}" pos="{self.f(self.P(pos))}" size="{size}" class="kp"/>')

    def nail(self, name, length, r):
        # flat box on the dorsal (+y) side near the tip; visual only
        pos = [0, r * 0.82, length * 0.62]
        self.w(f'<geom name="{self.pre}{name}" class="nail" type="box" pos="{self.f(self.P(pos))}" '
               f'size="{r * 0.62:.4g} {r * 0.22:.4g} {length * 0.30:.4g}"/>')

    # ------------------------------------------------------------------
    def emit(self):
        """Skeleton + collision capsules from the WebXR hand mesh; the visual is
        a linear-blend <skin> (emitted separately by skin_xml)."""
        from . import handmesh as hm
        p = self.p
        B, KP = hm.skeleton(p.scale, p.finger_len, p.palm_width)
        self.B, self.KP = B, KP
        L = hm.bone_lengths(B, KP)
        rad0, (plo, phi) = hm.fitted_radii()
        rr = p.radius * p.scale
        self.w(f'<body name="{self.pre}root" pos="{self.f(self.base_pos)}">')
        self.ind += 1
        for ax, nm in zip(np.eye(3), ["tx", "ty", "tz"]):
            lo, hi = ROOT_LIM[nm]
            self.w(f'<joint name="{self.pre}root_{nm}" class="root_slide" type="slide" axis="{self.f(self.P(ax))}" range="{lo} {hi}"/>')
        for ax, nm in zip([[0, 0, 1], [1, 0, 0], [0, 1, 0]], ["roll", "pitch", "yaw"]):
            self.hinge(f"root_{nm}", ax, ROOT_LIM[nm], cls="root_hinge")
        # forearm (visual) and palm collision box
        self.forearm_asset = self._forearm_mesh()
        self.w(f'<geom name="{self.pre}forearm" class="skin_vis" type="mesh" mesh="{self.pre}forearm_mesh"/>')
        c = (plo + phi) / 2 * np.array([p.palm_width, 1, 1]) * p.scale
        h = (phi - plo) / 2 * np.array([p.palm_width, p.radius, 1]) * p.scale
        self.w(f'<geom name="{self.pre}palm" class="col" type="box" pos="{self.f(self.P(c))}" size="{self.f(h)}"/>')
        self.w(f'<geom name="{self.pre}palm_heel" class="col" type="capsule" fromto="{self.f(self.P([c[0] - h[0] * 0.7, c[1], c[2] - h[2] * 0.8]))} {self.f(self.P([c[0] + h[0] * 0.7, c[1], c[2] - h[2] * 0.8]))}" size="{h[1]:.4g}"/>')
        self.site("kp_wrist", [0, 0, 0])
        self.site("palm_center", [c[0], c[1] - h[1], c[2]], size=0.003)
        self.site("palm_back", [c[0], c[1] + h[1], c[2]], size=0.003)
        Rw = {"root": np.eye(3)}
        Pw = {"root": np.zeros(3)}

        def open_body(name, parent):
            pos, R = B[name]
            Rp, pp = Rw[parent], Pw[parent]
            self.body_open(name, Rp.T @ (pos - pp), Rp.T @ R)
            Rw[name], Pw[name] = R, pos

        def seg_geoms(name, kpname, last=False):
            r = rad0[name] * rr
            Lb = L[name] - (r if last else 0)
            self.capsule(name, max(Lb, 0.002), r, cls="col")
            self.site(f"kp_{kpname}", [0, 0, 0])

        for fn in FINGERS:
            open_body(f"{fn}_prox", "root")
            self.hinge(f"{fn}_abd", [0, 1, 0], LIM["abd"] if fn in ("middle", "ring") else (-40, 25) if fn == "little" else (-25, 25))
            self.hinge(f"{fn}_mcp", [1, 0, 0], LIM["mcp"])
            seg_geoms(f"{fn}_prox", f"{fn}_mcp")
            open_body(f"{fn}_mid", f"{fn}_prox")
            self.hinge(f"{fn}_pip", [1, 0, 0], LIM["pip"])
            seg_geoms(f"{fn}_mid", f"{fn}_pip")
            open_body(f"{fn}_dist", f"{fn}_mid")
            self.hinge(f"{fn}_dip", [1, 0, 0], LIM["dip"])
            seg_geoms(f"{fn}_dist", f"{fn}_dip", last=True)
            self.site(f"kp_{fn}_tip", [0, 0, L[f"{fn}_dist"]])
            self.body_close(); self.body_close(); self.body_close()
        open_body("thumb_meta", "root")
        self.hinge("thumb_cmc_flex", [1, 0, 0], THUMB_LIM["cmc_flex"])
        self.hinge("thumb_cmc_abd", [0, 1, 0], THUMB_LIM["cmc_abd"])
        self.hinge("thumb_cmc_rot", [0, 0, 1], THUMB_LIM["cmc_rot"])
        seg_geoms("thumb_meta", "thumb_cmc")
        open_body("thumb_prox", "thumb_meta")
        self.hinge("thumb_mcp", [1, 0, 0], THUMB_LIM["mcp"])
        seg_geoms("thumb_prox", "thumb_mcp")
        open_body("thumb_dist", "thumb_prox")
        self.hinge("thumb_ip", [1, 0, 0], THUMB_LIM["ip"])
        seg_geoms("thumb_dist", "thumb_ip", last=True)
        self.site("kp_thumb_tip", [0, 0, L["thumb_dist"]])
        self.body_close(); self.body_close(); self.body_close()
        self.ind -= 1
        self.w("</body>")
        return self.lines

    def _forearm_mesh(self, n=28):
        """Elliptic, slightly tapered tube matching the mesh's wrist section."""
        p = self.p
        sc, rr = p.scale, p.radius
        rings = [(0.004, 0.98), (-0.006, 1.00), (-0.03, 1.04), (-0.07, 1.12), (-0.11, 1.18), (-0.15, 1.22)]
        cx, cy, ax, ay = -0.0015 * p.palm_width, 0.0035 * rr, 0.0262 * p.palm_width, 0.0172 * rr
        V = []
        for z, k in rings:
            for i in range(n):
                t = 2 * np.pi * i / n
                V.append([sc * (cx + k * ax * np.cos(t)), sc * (cy + k * ay * np.sin(t)), sc * z])
        V.append([sc * cx, sc * cy, sc * rings[-1][0]])
        F = []
        for r in range(len(rings) - 1):
            for i in range(n):
                a, b = r * n + i, r * n + (i + 1) % n
                c, d = a + n, b + n
                F += [[a, c, b], [b, c, d]]
        cap = len(V) - 1
        base = (len(rings) - 1) * n
        for i in range(n):
            F.append([base + i, cap, base + (i + 1) % n])
        V = np.array(V) @ self.M.T
        F = np.array(F)
        if self.side == "right":
            F = F[:, ::-1]
        return (f'<mesh name="{self.pre}forearm_mesh" vertex="{" ".join(f"{x:.5f}" for x in V.ravel())}" '
                f'face="{" ".join(str(int(i)) for i in F.ravel())}"/>')

    def skin_xml(self):
        from . import handmesh as hm
        V, F, groups, _, _ = hm.skin(self.B, self.p.radius)
        Vw = self.base_pos + V @ self.M.T
        if self.side == "left":
            F = F[:, ::-1]
        out = [f'    <skin name="{self.pre}skin" material="skin" inflate="0.0003" vertex="{" ".join(f"{x:.5f}" for x in Vw.ravel())}" '
               f'face="{" ".join(str(int(i)) for i in F.ravel())}">']
        for b, (ids, w) in groups.items():
            pos, R = self.B[b]
            bp = self.base_pos + self.M @ pos
            bq = mat2quat(self.M @ R @ self.M)
            out.append(f'      <bone body="{self.pre}{b}" bindpos="{self.f(bp)}" bindquat="{self.f(bq)}" '
                       f'vertid="{" ".join(str(int(i)) for i in ids)}" vertweight="{" ".join(f"{x:.4f}" for x in w)}"/>')
        out.append("    </skin>")
        return "\n".join(out)


KEYPOINTS = ["wrist"] + [f"thumb_{k}" for k in ["cmc", "mcp", "ip", "tip"]] + \
    [f"{f}_{k}" for f in FINGERS for k in ["mcp", "pip", "dip", "tip"]]


def build_mjcf(hands=("right",), params: dict | None = None, img_w=384, img_h=384) -> str:
    """Return an MJCF string with one or two hands.

    hands: ("right",) -> single right hand at the origin, joint names unprefixed.
           ("right","left") -> two hands, prefixes "R_" and "L_", placed at x=+-0.11.
    """
    params = params or {}
    two = len(hands) == 2
    body, skins, assets = [], [], []
    for side in hands:
        pre = ("R_" if side == "right" else "L_") if two else ""
        base = [-0.11 if side == "right" else 0.11, 0, 0] if two else [0, 0, 0]  # dancer facing the camera
        em = _Emitter(side, params.get(side, HandParams()), pre, base)
        body += em.emit()
        skins.append(em.skin_xml())
        assets.append("    " + em.forearm_asset)

    acts = []
    for side in hands:
        pre = ("R_" if side == "right" else "L_") if two else ""
        for j in root_joint_names(pre):
            cls = "a_slide" if j.endswith(("tx", "ty", "tz")) else "a_root"
            acts.append(f'    <position name="a_{j}" joint="{j}" class="{cls}"/>')
        for j in finger_joint_names(pre):
            acts.append(f'    <position name="a_{j}" joint="{j}" class="a_finger"/>')

    skin = " ".join(str(x) for x in params.get("right", HandParams()).skin_rgba)
    zc = 0.07
    dist = 0.62 if two else 0.46
    cams = f"""
    <camera name="front" pos="0 {-dist} {zc}" xyaxes="1 0 0 0 0 1" fovy="35"/>
    <camera name="side" pos="{dist} 0 {zc}" xyaxes="0 1 0 0 0 1" fovy="35"/>
    <camera name="top" pos="0 -0.02 {zc + dist}" xyaxes="1 0 0 0 1 0" fovy="35"/>
    <camera name="three_quarter" pos="{0.7 * dist:.4f} {-0.7 * dist:.4f} {zc + 0.12}" xyaxes="0.7071 0.7071 0 -0.13 0.13 0.98" fovy="35"/>
    <camera name="back" pos="0 {dist} {zc}" xyaxes="-1 0 0 0 0 1" fovy="35"/>"""
    return f"""<mujoco model="hastabench_{'two' if two else 'one'}_hand">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast" cone="elliptic">
    <flag warmstart="enable"/>
  </option>
  <visual>
    <global offwidth="{img_w}" offheight="{img_h}"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.40 0.40 0.40" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.80 0.83 0.87" rgb2="0.55 0.58 0.63" width="64" height="64"/>
    <material name="skin" rgba="{skin}" specular="0.25" shininess="0.35" reflectance="0"/>
{chr(10).join(assets)}
    <material name="nail" rgba="0.97 0.85 0.82 1" specular="0.4" shininess="0.6"/>
  </asset>
  <default>
    <joint limited="true" armature="0.0005" damping="0.03"/>
    <geom density="1000" friction="0.8 0.02 0.001" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <default class="finger"><joint/></default>
    <default class="root_hinge"><joint armature="0.003" damping="0.25"/></default>
    <default class="root_slide"><joint armature="0.3" damping="20"/></default>
    <default class="skin"><geom material="skin" contype="1" conaffinity="1" group="0"/></default>
    <default class="skin_vis"><geom material="skin" contype="0" conaffinity="0" mass="0" group="0"/></default>
    <default class="col"><geom contype="1" conaffinity="1" group="3" rgba="0.2 0.6 0.9 0.4"/></default>
    <default class="nail"><geom material="nail" contype="0" conaffinity="0" mass="0" group="0"/></default>
    <default class="kp"><site group="4" rgba="1 0 0 1"/></default>
    <default class="a_finger"><position kp="0.5" forcerange="-0.35 0.35"/></default>
    <default class="a_root"><position kp="4" forcerange="-20 20"/></default>
    <default class="a_slide"><position kp="300" forcerange="-200 200"/></default>
  </default>
  <worldbody>
    <light name="key" pos="0.25 -0.5 0.6" dir="-0.35 0.7 -0.6" diffuse="0.55 0.55 0.55" castshadow="false"/>
    <light name="fill" pos="-0.4 -0.3 0.3" dir="0.6 0.5 -0.3" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <light name="rim" pos="0.1 0.6 0.4" dir="-0.1 -0.8 -0.4" diffuse="0.3 0.3 0.3" castshadow="false"/>
{cams}
{chr(10).join(body)}
  </worldbody>
  <deformable>
{chr(10).join(skins)}
  </deformable>
  <contact>
{_excludes(hands, two)}
  </contact>
  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
"""


def _excludes(hands, two):
    # palm vs. thumb proximal phalanx sits on the palm surface -> exclude
    out = []
    for side in hands:
        pre = ("R_" if side == "right" else "L_") if two else ""
        out.append(f'    <exclude body1="{pre}root" body2="{pre}thumb_prox"/>')
        out.append(f'    <exclude body1="{pre}index_prox" body2="{pre}thumb_meta"/>')
    if two:
        # the thenar (thumb-base) capsules over-approximate soft tissue that compresses when
        # palms are pressed together (Anjali); do not let them collide across hands
        out.append('    <exclude body1="R_thumb_meta" body2="L_thumb_meta"/>')
    return "\n".join(out)
