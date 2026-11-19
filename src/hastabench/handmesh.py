"""Natural human-hand skin mesh: WebXR 'generic-hand' (MIT, (c) 2019 Amazon,
immersive-web/webxr-input-profiles).  We read the rigged glTF, express it in
the HastaBench hand frame (+z distal, -y palmar, +x radial), and derive the
kinematic skeleton (joint positions, bone frames) from its bind pose, so the
MuJoCo bodies drive the mesh through linear-blend skinning (<skin>).
"""
from __future__ import annotations

import functools
import os

import numpy as np

GLB = os.path.join(os.path.dirname(__file__), "..", "..", "ext_assets", "webxr", "package", "dist",
                   "profiles", "generic-hand", "right.glb")

FMAP = {"index": "index-finger", "middle": "middle-finger", "ring": "ring-finger", "little": "pinky-finger"}

# glTF joint -> our body.  Metacarpals of the four fingers are rigid with the palm.
def body_of(jname: str) -> str:
    if jname == "wrist":
        return "root"
    if jname.startswith("thumb"):
        return {"thumb-metacarpal": "thumb_meta", "thumb-phalanx-proximal": "thumb_prox",
                "thumb-phalanx-distal": "thumb_dist", "thumb-tip": "thumb_dist"}[jname]
    for f, g in FMAP.items():
        if jname.startswith(g):
            seg = jname[len(g) + 1:]
            return {"metacarpal": "root", "phalanx-proximal": f"{f}_prox", "phalanx-intermediate": f"{f}_mid",
                    "phalanx-distal": f"{f}_dist", "tip": f"{f}_dist"}[seg]
    raise KeyError(jname)


def _norm(v):
    return v / np.linalg.norm(v)


def _frame(z, xhint):
    z = _norm(z)
    x = _norm(xhint - z * (xhint @ z))
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)          # columns = axes


@functools.lru_cache(maxsize=1)
def load():
    from pygltflib import GLTF2
    g = GLTF2().load(GLB)
    blob = g.binary_blob()

    def acc(i):
        a = g.accessors[i]
        bv = g.bufferViews[a.bufferView]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a.type]
        dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a.componentType]
        return np.frombuffer(blob[bv.byteOffset + (a.byteOffset or 0):], dtype=dt, count=a.count * nc).reshape(a.count, nc).astype(float)

    sk = g.skins[0]
    names = [g.nodes[j].name for j in sk.joints]
    ib = acc(sk.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
    J = {n: np.linalg.inv(m)[:3, 3] for n, m in zip(names, ib)}
    pr = g.meshes[0].primitives[0]
    V = acc(pr.attributes.POSITION)
    Fc = acc(pr.indices).astype(int).reshape(-1, 3)
    JI = acc(pr.attributes.JOINTS_0).astype(int)
    W = acc(pr.attributes.WEIGHTS_0)

    # hand frame
    wrist = J["wrist"]
    # z = middle-finger direction (rest), blended with wrist->MCP so the palm is upright
    z_h = _norm(_norm(J["middle-finger-phalanx-intermediate"] - J["middle-finger-phalanx-proximal"]) * 0.75
                + _norm(J["middle-finger-phalanx-proximal"] - wrist) * 0.25)
    x_h = J["index-finger-phalanx-proximal"] - J["pinky-finger-phalanx-proximal"]
    Rh = _frame(z_h, x_h)                  # columns: hand axes in glTF coords
    to_h = lambda p: (np.asarray(p) - wrist) @ Rh
    Jh = {n: to_h(p) for n, p in J.items()}
    Vh = to_h(V)
    # sanity: thumb lies on the palmar (-y) side for a right hand
    assert Jh["thumb-phalanx-distal"][1] < 0.0, "unexpected handedness"
    return dict(J=Jh, V=Vh, F=Fc, JI=JI, W=W, names=names)


def skeleton(scale=1.0, finger_len=None, palm_width=1.0):
    """Bind-pose body frames in the hand frame for (optionally) re-proportioned
    hands.  Returns {body: (pos, R)} plus keypoint positions and tip offsets."""
    d = load()
    J = d["J"]
    fl = finger_len or {}
    B, KP = {}, {}
    B["root"] = (np.zeros(3), np.eye(3))
    KP["wrist"] = np.zeros(3)
    xh = np.array([1.0, 0, 0])

    def chain(prefix, pts, lens_scale, base_scale_x):
        # pts: list of glTF joint positions along the chain (base ... tip)
        P = [pts[0] * np.array([base_scale_x, 1, 1]) * scale]
        for a, b in zip(pts[:-1], pts[1:]):
            P.append(P[-1] + (b - a) * scale * lens_scale)
        return P

    for f, g in FMAP.items():
        pts = [J[f"{g}-phalanx-proximal"], J[f"{g}-phalanx-intermediate"], J[f"{g}-phalanx-distal"], J[f"{g}-tip"]]
        P = chain(f, pts, fl.get(f, 1.0), palm_width)
        for seg, a, b in zip(["prox", "mid", "dist"], P[:-1], P[1:]):
            B[f"{f}_{seg}"] = (a, _frame(b - a, xh))
        for k, p in zip(["mcp", "pip", "dip", "tip"], P):
            KP[f"{f}_{k}"] = p
    pts = [J["thumb-metacarpal"], J["thumb-phalanx-proximal"], J["thumb-phalanx-distal"], J["thumb-tip"]]
    P = chain("thumb", pts, fl.get("thumb", 1.0), palm_width)
    # thumb flexion axis: positive flexion curls toward the palm centre / fingers
    for seg, a, b in zip(["meta", "prox", "dist"], P[:-1], P[1:]):
        z = _norm(b - a)
        pad = _norm(0.8 * np.array([-1.0, 0, 0]) + 0.6 * np.array([0, -1.0, 0]))
        pad = _norm(pad - z * (pad @ z))
        y = -pad
        x = np.cross(y, z)
        B[f"thumb_{seg}"] = (a, np.stack([x, y, z], 1))
    for k, p in zip(["cmc", "mcp", "ip", "tip"], P):
        KP[f"thumb_{k}"] = p
    return B, KP


def bone_lengths(B, KP):
    L = {}
    for f in ["index", "middle", "ring", "little"]:
        L[f"{f}_prox"] = np.linalg.norm(KP[f"{f}_pip"] - KP[f"{f}_mcp"])
        L[f"{f}_mid"] = np.linalg.norm(KP[f"{f}_dip"] - KP[f"{f}_pip"])
        L[f"{f}_dist"] = np.linalg.norm(KP[f"{f}_tip"] - KP[f"{f}_dip"])
    L["thumb_meta"] = np.linalg.norm(KP["thumb_mcp"] - KP["thumb_cmc"])
    L["thumb_prox"] = np.linalg.norm(KP["thumb_ip"] - KP["thumb_mcp"])
    L["thumb_dist"] = np.linalg.norm(KP["thumb_tip"] - KP["thumb_ip"])
    return L


def skin(B_new, radius=1.0):
    """Vertices re-skinned onto (possibly re-proportioned) bind frames via LBS
    from the nominal skeleton.  Returns V (n,3), F, per-body (vertid, weight)."""
    d = load()
    B0, _ = skeleton()
    V, JI, W = d["V"], d["JI"], d["W"]
    bodies = list(B0)
    bidx = {b: i for i, b in enumerate(bodies)}
    jb = np.array([bidx[body_of(n)] for n in d["names"]])
    nV = len(V)
    Wb = np.zeros((nV, len(bodies)))
    for k in range(4):
        np.add.at(Wb, (np.arange(nV), jb[JI[:, k]]), W[:, k])
    Wb /= Wb.sum(1, keepdims=True)
    Vn = np.zeros_like(V)
    S = np.diag([radius, radius, 1.0])
    for b, i in bidx.items():
        p0, R0 = B0[b]
        p1, R1 = B_new[b]
        loc = (V - p0) @ R0                  # vertex in bone frame
        if b != "root":
            loc = loc @ S
        Vn += Wb[:, [i]] * (p1 + loc @ R1.T)
    groups = {}
    for b, i in bidx.items():
        ids = np.nonzero(Wb[:, i] > 1e-4)[0]
        groups[b] = (ids, Wb[ids, i])
    return Vn, d["F"], groups, Wb, bodies


def fitted_radii():
    """Collision-capsule radius per bone and palm box from the mesh."""
    d = load()
    B, KP = skeleton()
    _, _, _, Wb, bodies = skin(B)
    L = bone_lengths(B, KP)
    V = d["V"]
    rad = {}
    for b in bodies:
        if b == "root":
            continue
        i = bodies.index(b)
        m = Wb[:, i] > 0.6
        p, R = B[b]
        loc = (V[m] - p) @ R
        sel = (loc[:, 2] > 0.15 * L[b]) & (loc[:, 2] < 0.85 * L[b])
        if sel.sum() < 5:
            sel = np.ones(len(loc), bool)
        rad[b] = float(np.median(np.linalg.norm(loc[sel, :2], axis=1)))
    m = Wb[:, 0] > 0.8
    P = V[m]
    P = P[(P[:, 2] > 0.005)]
    lo, hi = np.percentile(P, 6, 0), np.percentile(P, 94, 0)
    return rad, (lo, hi)
