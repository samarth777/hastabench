"""Hidden final-state scorer (shared by benchmark and RL reporting).

Metrics (all on the *settled* simulator state):
  kp_err_mm      mean 3-D error of 20 hand keypoints (MCP/PIP/DIP/tip of each
                 finger, thumb CMC/MCP/IP/tip), expressed in the hand's own
                 frame, so translation and global rotation do not matter.
  finger_err_mm  per-finger max keypoint error (a single wrong finger fails).
  orient_err_deg geodesic angle between achieved and target hand orientation.
  contacts       fraction of required contacts satisfied (surface gap <= 5 mm)
                 and of 'apart' constraints satisfied (gap >= 8 mm).
  penetration_mm max interpenetration; valid if <= 3 mm.
Success = kp_err <= KP_TOL and every finger_err <= FINGER_TOL and
orient <= ORIENT_TOL and all constraints satisfied and valid.
"""
from __future__ import annotations

import numpy as np

KP_TOL_MM = 10.0
FINGER_TOL_MM = 20.0
ORIENT_TOL_DEG = 20.0
EVAL_TOUCH_MM = 5.0
EVAL_APART_MM = 8.0
PEN_TOL_MM = 3.0

FINGER_SLICES = {"thumb": slice(1, 5), "index": slice(5, 9), "middle": slice(9, 13),
                 "ring": slice(13, 17), "little": slice(17, 21)}


def rot_angle_deg(Ra, Rb):
    R = np.asarray(Ra).T @ np.asarray(Rb)
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def score_hand(kp, R, kp_t, R_t, contact_d, apart_d, pen_m, scale=1.0):
    """kp, kp_t: 21x3 in hand frame (m).  scale: target hand size / nominal,
    so tolerances scale with hand proportions."""
    kp, kp_t = np.asarray(kp), np.asarray(kp_t)
    e = np.linalg.norm(kp - kp_t, axis=1) * 1000.0 / scale
    kp_err = float(e[1:].mean())
    fe = {f: float(e[s].max()) for f, s in FINGER_SLICES.items()}
    orient = rot_angle_deg(R, R_t)
    c_ok = [d * 1000 <= EVAL_TOUCH_MM for d in contact_d]
    a_ok = [d * 1000 >= EVAL_APART_MM for d in apart_d]
    cons = c_ok + a_ok
    cons_frac = float(np.mean(cons)) if cons else 1.0
    pen = pen_m * 1000
    valid = pen <= PEN_TOL_MM
    success = (kp_err <= KP_TOL_MM and max(fe.values()) <= FINGER_TOL_MM and orient <= ORIENT_TOL_DEG
               and all(cons) and valid)
    # continuous score in [0,1] for leaderboards / partial credit
    hs = (0.6 * np.exp(-kp_err / 15.0) + 0.2 * max(0.0, 1 - orient / 45.0) + 0.2 * cons_frac) * (1.0 if valid else 0.5)
    return dict(success=bool(success), hasta_score=float(hs), kp_err_mm=kp_err, finger_err_mm=fe,
                orient_err_deg=orient, contacts_ok=[bool(x) for x in c_ok], apart_ok=[bool(x) for x in a_ok],
                constraint_frac=cons_frac, penetration_mm=float(pen), valid=bool(valid))


def score_sim(sim, target, scale=1.0):
    """Score the current state of a single-hand HandSim against a target record."""
    p = sim.prefixes[0]
    cd = [sim.geom_dist(p + a, p + b) for a, b in target["contacts"]]
    ad = [sim.geom_dist(p + a, p + b) for a, b in target["apart"]]
    return score_hand(sim.keypoints(p), sim.hand_rot(p), target["keypoints_hand"], target["hand_rot"],
                      cd, ad, sim.penetration(), scale)


def score_two(sim, target, scale=1.0):
    """Two-hand (samyuta) scoring: per-hand shape+orientation, plus relative
    pose of the hands and cross-hand contacts."""
    out = {}
    hand_ok, hs = [], []
    for p in sim.prefixes:
        t = target["hands"][p]
        cd = [sim.geom_dist(a, b) for a, b in t["contacts"]]
        ad = [sim.geom_dist(a, b) for a, b in t["apart"]]
        r = score_hand(sim.keypoints(p), sim.hand_rot(p), t["keypoints_hand"], t["hand_rot"], cd, ad, 0.0, scale)
        out[p] = r
        hand_ok.append(r["success"])
        hs.append(r["hasta_score"])
    # relative pose: right hand expressed in left hand frame
    Rr, Rl = sim.hand_rot("R_"), sim.hand_rot("L_")
    rel_R = Rl.T @ Rr
    rel_p = Rl.T @ (sim.hand_pos("R_") - sim.hand_pos("L_"))
    rel_R_t = np.asarray(target["rel_rot"])
    rel_p_t = np.asarray(target["rel_pos"])
    rel_ang = rot_angle_deg(rel_R, rel_R_t)
    rel_pos_mm = float(np.linalg.norm(rel_p - rel_p_t) * 1000 / scale)
    cd = [sim.geom_dist(a, b) for a, b in target["cross_contacts"]]
    ad = [sim.geom_dist(a, b) for a, b in target["cross_apart"]]
    c_ok = [d * 1000 <= EVAL_TOUCH_MM for d in cd] + [d * 1000 >= EVAL_APART_MM for d in ad]
    pen = sim.penetration() * 1000
    orient = rot_angle_deg(sim.hand_rot("L_"), target["hands"]["L_"]["hand_rot"])
    success = all(hand_ok) and rel_ang <= ORIENT_TOL_DEG and rel_pos_mm <= 25.0 and all(c_ok) and pen <= PEN_TOL_MM
    cf = float(np.mean(c_ok)) if c_ok else 1.0
    score = (0.5 * float(np.mean(hs)) + 0.2 * max(0, 1 - rel_ang / 45) + 0.1 * np.exp(-rel_pos_mm / 30) + 0.2 * cf)
    out.update(success=bool(success), hasta_score=float(score) * (1.0 if pen <= PEN_TOL_MM else 0.5),
               rel_rot_err_deg=rel_ang, rel_pos_err_mm=rel_pos_mm, cross_contacts_ok=[bool(x) for x in c_ok],
               penetration_mm=float(pen), valid=bool(pen <= PEN_TOL_MM), left_orient_err_deg=orient,
               kp_err_mm=float(np.mean([out[p]["kp_err_mm"] for p in sim.prefixes])),
               orient_err_deg=float(np.mean([out[p]["orient_err_deg"] for p in sim.prefixes])))
    return out
