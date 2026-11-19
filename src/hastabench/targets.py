"""Build the versioned target dataset.

For every hasta: intent pose -> constrained least-squares (contacts touch,
'apart' pairs separated, no interpenetration, joint limits) -> physics
settle in the simulator -> automatic validation -> save json + images.

python -m hastabench.targets --out data/targets_v1
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
from scipy.optimize import least_squares
from PIL import Image

from .sim import HandSim
from .model import DEG, finger_joint_names, KEYPOINTS
from .hastas import HASTAS, DISPLAY, CORE8, ALL28

TOUCH_TOL = 0.0045     # required contact: surfaces within 4.5 mm (eval tolerance is 5 mm)
APART_MIN = 0.010      # 'apart' pairs must be at least 1 cm apart
PEN_TOL = 0.002        # max allowed interpenetration (m)

COLLIDE_GEOMS = ["palm"] + [f"{f}_{s}" for f in ["thumb", "index", "middle", "ring", "little"]
                            for s in (["meta", "prox", "dist"] if f == "thumb" else ["prox", "mid", "dist"])]


def collision_pairs(sim: HandSim, prefix=""):
    """Geom pairs that can collide (exclude same/parent-child bodies and the
    explicit <exclude> list), mirroring MuJoCo's own filtering."""
    m = sim.m
    ex = set()
    for i in range(m.nexclude if hasattr(m, "nexclude") else 0):
        pass
    excl = {(m.body(f"{prefix}root").id, m.body(f"{prefix}thumb_prox").id),
            (m.body(f"{prefix}index_prox").id, m.body(f"{prefix}thumb_meta").id)}
    pairs = []
    for a, b in itertools.combinations(COLLIDE_GEOMS, 2):
        ga, gb = m.geom(prefix + a), m.geom(prefix + b)
        ba, bb = int(ga.bodyid[0]), int(gb.bodyid[0])
        if ba == bb or m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
            continue
        if (ba, bb) in excl or (bb, ba) in excl:
            continue
        pairs.append((prefix + a, prefix + b))
    return pairs


def solve_pose(sim: HandSim, intent_deg: dict, contacts, apart, root_q=None, prefix="", w_intent=1.0,
               fixed_q=None, pairs=None, extra_pairs=()):
    """Return a full q (internal units) close to the intent that satisfies
    the contact constraints.  Only this hand's finger joints are optimised."""
    names = finger_joint_names(prefix)
    idx = np.array([sim.joint_names.index(n) for n in names])
    q0 = sim.q() if fixed_q is None else np.array(fixed_q, float)
    if root_q is not None:
        ridx = [sim.joint_names.index(prefix + "root_" + k) for k in ["tx", "ty", "tz", "roll", "pitch", "yaw"]]
        q0[ridx] = [root_q[0], root_q[1], root_q[2], root_q[3] * DEG, root_q[4] * DEG, root_q[5] * DEG]
    x_int = np.array([intent_deg.get(n[len(prefix):], 0.0) * DEG for n in names])
    lo, hi = sim.lo[idx], sim.hi[idx]
    x_int = np.clip(x_int, lo + 1e-4, hi - 1e-4)
    pairs = pairs if pairs is not None else collision_pairs(sim, prefix)
    pairs = list(pairs) + list(extra_pairs)
    cset = {tuple(sorted(c)) for c in contacts}

    def res(x):
        q = q0.copy()
        q[idx] = x
        sim.set_kinematic(q)
        r = [w_intent * (x - x_int)]
        r.append(np.array([300.0 * max(0.0, sim.geom_dist(a, b) - 0.0005) for a, b in contacts]))
        r.append(np.array([300.0 * max(0.0, APART_MIN + 0.004 - sim.geom_dist(a, b)) for a, b in apart]))
        pen = []
        for a, b in pairs:
            if tuple(sorted((a, b))) in cset:
                pen.append(60.0 * max(0.0, -sim.geom_dist(a, b, 0.01) - 0.0005))
            else:
                pen.append(60.0 * max(0.0, -sim.geom_dist(a, b, 0.01)))
        r.append(np.array(pen))
        return np.concatenate([np.atleast_1d(v) for v in r])

    best = None
    for k, start in enumerate([x_int, np.clip(x_int * 0.7, lo, hi), np.clip(x_int + 0.15, lo, hi)]):
        sol = least_squares(res, start, bounds=(lo, hi), diff_step=1e-3, max_nfev=400, xtol=1e-8, ftol=1e-10)
        if best is None or sol.cost < best.cost:
            best = sol
        if best.cost < 1e-6:
            break
    q = q0.copy()
    q[idx] = best.x
    return q, best.cost


def check_constraints(sim: HandSim, contacts, apart):
    cd = {f"{a}|{b}": float(sim.geom_dist(a, b)) for a, b in contacts}
    ad = {f"{a}|{b}": float(sim.geom_dist(a, b)) for a, b in apart}
    ok = all(v <= TOUCH_TOL for v in cd.values()) and all(v >= APART_MIN for v in ad.values())
    return ok, cd, ad


def build_target(sim: HandSim, name: str, out_dir: str, views=("front", "side", "three_quarter", "top", "back")):
    meaning, intent, contacts, apart = HASTAS[name]
    roll, pitch, yaw = DISPLAY[name]
    root = (0, 0, 0, roll, pitch, yaw)
    sim.reset()
    q_k, cost = solve_pose(sim, intent, contacts, apart, root_q=root)
    # physics settle from the kinematic solution
    sim.set_kinematic(q_k)
    q_s = sim.command(q_k, steps=400)
    ok, cd, ad = check_constraints(sim, contacts, apart)
    pen = sim.penetration()
    valid = ok and pen <= PEN_TOL
    rec = dict(
        id=name, meaning=meaning, version="v1", hands=["right"],
        joint_names=sim.joint_names,
        q=q_s.tolist(), q_command=q_k.tolist(), q_deg=sim.to_human(q_s).round(2).tolist(),
        keypoints_hand=sim.keypoints(frame="hand").tolist(),
        hand_rot=sim.hand_rot().tolist(), hand_pos=sim.hand_pos().tolist(),
        contacts=[list(c) for c in contacts], apart=[list(c) for c in apart],
        contact_dist=cd, apart_dist=ad, penetration=pen, solve_cost=float(cost),
        valid=bool(valid), core8=name in CORE8, keypoint_names=KEYPOINTS,
        images={v: f"{name}_{v}.png" for v in views},
    )
    os.makedirs(out_dir, exist_ok=True)
    for v in views:
        Image.fromarray(sim.render(v)).save(os.path.join(out_dir, f"{name}_{v}.png"))
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/targets_v1")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    sim = HandSim(img=384)
    names = args.only or ALL28
    summary = []
    for n in names:
        t = time.time()
        r = build_target(sim, n, args.out)
        summary.append(dict(id=n, valid=r["valid"], pen_mm=round(1000 * r["penetration"], 2),
                            contacts_mm={k: round(1000 * v, 1) for k, v in r["contact_dist"].items()},
                            apart_mm={k: round(1000 * v, 1) for k, v in r["apart_dist"].items()},
                            cost=round(r["solve_cost"], 5), sec=round(time.time() - t, 1)))
        print(json.dumps(summary[-1]), flush=True)
    # contact sheet
    ims = []
    for n in names:
        a = np.array(Image.open(os.path.join(args.out, f"{n}_front.png")).resize((192, 192)))
        b = np.array(Image.open(os.path.join(args.out, f"{n}_side.png")).resize((192, 192)))
        ims.append(np.concatenate([a, b], 1))
    cols = 4
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]) + 255)
    rows = [np.concatenate(ims[i:i + cols], 1) for i in range(0, len(ims), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.out, "contact_sheet.png"))
    if args.only is None:
        with open(os.path.join(args.out, "index.json"), "w") as f:
            json.dump(dict(version="v1", core8=CORE8, all28=ALL28, summary=summary), f, indent=1)


if __name__ == "__main__":
    main()
