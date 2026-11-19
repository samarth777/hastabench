"""Two-hand (samyuta) targets: reuse the validated single-hand shapes, then
solve for both hands' root poses so the cross-hand contacts hold without
interpenetration; settle and validate.

python -m hastabench.samyuta --out data/samyuta_v1
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from .sim import HandSim
from .model import finger_joint_names, root_joint_names, DEG, KEYPOINTS
from .hastas import SAMYUTA, SAMYUTA_OVERRIDE
from .core import load_targets
from .targets import COLLIDE_GEOMS, TOUCH_TOL, APART_MIN, PEN_TOL


def build(sim: HandSim, name, single, out_dir):
    meaning, rs, ls, rroot, lroot, cc, ca, views = SAMYUTA[name]
    q = np.zeros(sim.n)
    for p, shape, root in (("R_", rs, rroot), ("L_", ls, lroot)):
        src = single[shape]
        for j in finger_joint_names():
            q[sim.joint_names.index(p + j)] = src["q_command"][src["joint_names"].index(j)]
        for k, v in SAMYUTA_OVERRIDE.get(name, {}).items():
            q[sim.joint_names.index(p + k)] = v * DEG
        for k, v in zip(["tx", "ty", "tz", "roll", "pitch", "yaw"], root):
            q[sim.joint_names.index(p + "root_" + k)] = v if k.startswith("t") else v * DEG
    # free joints in the cross-hand solve: thumbs always; for mirror-symmetric gestures also the
    # finger knuckles (MCP), so palms and fingertips can meet together
    sym0 = rs == ls and tuple(rroot) == tuple(lroot)
    free = lambda j: "thumb" in j or (sym0 and j.endswith("_mcp") and "thumb" not in j)
    thumbs_r = [j for j in finger_joint_names("R_") if free(j)]
    thumbs_l = [j for j in finger_joint_names("L_") if free(j)]
    ridx = np.array([sim.joint_names.index(n) for n in root_joint_names("R_") + root_joint_names("L_") + thumbs_r + thumbs_l])
    scale_full = np.array([20, 20, 20, 10, 10, 10] * 2 + [0.5] * (len(thumbs_r) + len(thumbs_l)), float)
    # mirror-symmetric samyutas share ONE parameter set for both hands (joint values are
    # mirror-symmetric by construction), so the pose is exactly symmetric about the midline
    sym = rs == ls and tuple(rroot) == tuple(lroot)
    nr, nt = 6, len(thumbs_r)
    if sym:
        sel = np.r_[0:nr, 2 * nr:2 * nr + nt]
        expand = lambda x: np.r_[x[:nr], x[:nr], x[nr:], x[nr:]]
    else:
        sel = np.arange(len(ridx))
        expand = lambda x: x
    scale = scale_full[sel]
    x0 = q[ridx][sel].copy()
    G = COLLIDE_GEOMS + ["palm_heel"]
    cross = [("R_" + a, "L_" + b) for a, b in itertools.product(G, G) if not (a == b == "thumb_meta")]
    cset = {tuple(c) for c in cc}

    def res(x):
        qq = q.copy(); qq[ridx] = expand(x)
        sim.set_kinematic(qq)
        r = [(x - x0) * scale * 0.3]
        r.append([300 * max(0, sim.geom_dist(a, b) - 0.0005) for a, b in cc])
        r.append([300 * max(0, APART_MIN + 0.004 - sim.geom_dist(a, b)) for a, b in ca])
        r.append([300 * max(0, -sim.geom_dist(a, b, 0.01) - (0.0005 if (a, b) in cset else 0)) for a, b in cross])
        return np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in r])

    lo, hi = sim.lo[ridx][sel], sim.hi[ridx][sel]
    best = None
    rng = np.random.default_rng(0)
    for k in range(4):
        start = np.clip(x0 + (0 if k == 0 else rng.normal(0, 0.01, len(x0)) / scale * 20), lo, hi)
        sol = least_squares(res, start, bounds=(lo, hi), diff_step=1e-3, max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
    print(name, "solve cost", round(float(best.cost), 4), "symmetric" if sym else "", flush=True)
    q[ridx] = expand(best.x)
    sim.set_kinematic(q)
    sim.command(q, steps=400)
    cd = {f"{a}|{b}": float(sim.geom_dist(a, b)) for a, b in cc}
    ad = {f"{a}|{b}": float(sim.geom_dist(a, b)) for a, b in ca}
    pen = sim.penetration()
    hands = {}
    for p, shape in (("R_", rs), ("L_", ls)):
        src = single[shape]
        hands[p] = dict(shape=shape, keypoints_hand=sim.keypoints(p).tolist(), hand_rot=sim.hand_rot(p).tolist(),
                        contacts=[[p + a, p + b] for a, b in src["contacts"]], apart=[[p + a, p + b] for a, b in src["apart"]])
    # per-hand constraints the cross-hand solve broke (e.g. Pataka thumbs in
    # Anjali rest against each other) are dropped from that hand's target
    for h in hands.values():
        h["contacts"] = [c for c in h["contacts"] if sim.geom_dist(*c) <= TOUCH_TOL]
        h["apart"] = [c for c in h["apart"] if sim.geom_dist(*c) >= APART_MIN]
    own_ok = True
    valid = all(v <= TOUCH_TOL for v in cd.values()) and all(v >= APART_MIN for v in ad.values()) and pen <= PEN_TOL and own_ok
    Rr, Rl = sim.hand_rot("R_"), sim.hand_rot("L_")
    rec = dict(id=name, meaning=meaning, version="v1.1", hands=["right", "left"], joint_names=sim.joint_names,
               q=sim.q().tolist(), q_command=q.tolist(), q_deg=sim.to_human(sim.q()).round(2).tolist(),
               hands_detail=None, rel_rot=(Rl.T @ Rr).tolist(), rel_pos=(Rl.T @ (sim.hand_pos("R_") - sim.hand_pos("L_"))).tolist(),
               cross_contacts=[list(c) for c in cc], cross_apart=[list(c) for c in ca], cross_contact_dist=cd,
               cross_apart_dist=ad, penetration=pen, own_contacts_ok=own_ok, valid=bool(valid), keypoint_names=KEYPOINTS,
               images={v: f"{name}_{v}.png" for v in ("front", "side", "three_quarter", "top", "back")}, views=list(views))
    rec["hands"] = hands
    rec["hand_sides"] = ["right", "left"]
    os.makedirs(out_dir, exist_ok=True)
    for v in ("front", "side", "three_quarter", "top", "back"):
        Image.fromarray(sim.render(v)).save(os.path.join(out_dir, f"{name}_{v}.png"))
    json.dump(rec, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/samyuta_v1")
    args = ap.parse_args()
    single = load_targets()
    sim = HandSim(("right", "left"), img=384)
    sheet = []
    for n in SAMYUTA:
        r = build(sim, n, single, args.out)
        print(json.dumps(dict(id=n, valid=r["valid"], pen_mm=round(1000 * r["penetration"], 2), own=r["own_contacts_ok"],
                              cross_mm={k: round(1000 * v, 1) for k, v in r["cross_contact_dist"].items()},
                              apart_mm={k: round(1000 * v, 1) for k, v in r["cross_apart_dist"].items()})), flush=True)
        sheet.append(np.concatenate([np.array(Image.open(os.path.join(args.out, f"{n}_{v}.png")).resize((256, 256)))
                                     for v in ("front", "top", "three_quarter")], 1))
    Image.fromarray(np.concatenate(sheet, 0)).save(os.path.join(args.out, "contact_sheet.png"))


if __name__ == "__main__":
    main()
