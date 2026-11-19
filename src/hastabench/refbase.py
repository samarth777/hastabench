"""Reference baselines for the HastaBench report (all on the same start states).

  rest        submit the seeded start pose
  mean_shape  average finger configuration of the 28 hastas, at the target orientation
              (a 'featureless' prior; it uses the hidden orientation)
  cma_image   CMA-ES on silhouette IoU vs. the two target images (image-only, 2000 poses)
  cma_score   'blind' CMA-ES on the hidden score itself (2000 scored poses)
  ik_oracle   least squares on hidden 3-D keypoints (privileged upper bound)
"""
import argparse, json, os, time
import numpy as np
import cma
from PIL import Image
from .core import HastaCore, load_targets
from .hastas import ALL28
from .model import finger_joint_names, DEG
from .scorer import score_sim, score_two
from .bench import skin_mask

RES = 128


def opt_idx(core):
    s = core.sim
    names = []
    for p in s.prefixes:
        names += finger_joint_names(p) + [p + "root_roll", p + "root_pitch", p + "root_yaw"]
        if len(s.prefixes) == 2:
            names += [p + "root_tx", p + "root_ty", p + "root_tz"]
    return np.array([s.joint_names.index(n) for n in names])


def evaluate(core, q0, qt):
    s = core.sim
    s.reset(q0); s.command(q0)
    s.command(qt)


def score(core, t):
    return score_two(core.sim, t) if len(core.sim.prefixes) == 2 else score_sim(core.sim, t)


def run_cma(core, t, seed, objective, evals=2000):
    s = core.sim
    core.reset(t, seed)
    q0 = s.q()
    idx = opt_idx(core)
    lo, hi = s.lo[idx], s.hi[idx]
    views = ("front", "top") if len(s.prefixes) == 2 else ("front", "three_quarter")
    tm = [skin_mask(np.array(Image.open(os.path.join(t["_dir"], t["images"][v])).convert("RGB").resize((RES, RES)))) for v in views]
    x0 = (q0[idx] - lo) / (hi - lo)

    def f(x):
        q = q0.copy()
        q[idx] = lo + np.clip(x, 0, 1) * (hi - lo)
        evaluate(core, q0, q)
        if objective == "score":
            return -score(core, t)["hasta_score"]
        l = 0
        for v, m in zip(views, tm):
            c = skin_mask(s.render(v, size=RES))
            l += 1 - (c & m).sum() / max((c | m).sum(), 1)
        return l / len(views)
    es = cma.CMAEvolutionStrategy(x0, 0.2, {"bounds": [0, 1], "maxfevals": evals, "seed": seed + 1, "verbose": -9, "popsize": 16})
    best_x, best_f = x0, f(x0)
    while not es.stop():
        X = es.ask()
        F = [f(x) for x in X]
        es.tell(X, F)
        i = int(np.argmin(F))
        if F[i] < best_f:
            best_f, best_x = F[i], X[i]
    q = q0.copy(); q[idx] = lo + np.clip(best_x, 0, 1) * (hi - lo)
    evaluate(core, q0, q)
    return es.result.evaluations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--targets", default="all28")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", type=int, default=2000)
    a = ap.parse_args()
    if a.targets == "samyuta":
        T = load_targets("samyuta_v1"); names = [k for k, v in T.items() if v["valid"]]; core = HastaCore(hands=("right", "left"))
    else:
        T = load_targets(); names = ALL28; core = HastaCore()
    s = core.sim
    single = load_targets()
    fidx = [s.joint_names.index(j) for j in finger_joint_names()] if a.targets != "samyuta" else None
    os.makedirs("results/bench", exist_ok=True)
    out = f"results/bench/REF_{a.method}_{a.targets}_s{a.seed}.jsonl"
    with open(out, "w") as fo:
        for n in names:
            t = T[n]; t0 = time.time(); info = {}
            core.reset(t, a.seed)
            q0 = s.q()
            if a.method == "rest":
                pass
            elif a.method == "mean_shape":
                if fidx is None:
                    continue
                q = np.array(t["q_command"])
                q[fidx] = np.mean([np.array(single[k]["q_command"])[fidx] for k in ALL28], 0)
                evaluate(core, q0, q)
            elif a.method in ("cma_image", "cma_score"):
                info["evals"] = run_cma(core, t, a.seed, a.method.split("_")[1], a.evals)
            elif a.method == "ik_oracle":
                evaluate(core, q0, np.array(t["q_command"]))   # privileged: the solved target configuration
            sc = score(core, t)
            rec = dict(model="REF_" + a.method, target=n, seed=a.seed, score=round(100 * sc["hasta_score"], 2),
                       raw_score=round(100 * sc["hasta_score"], 2), solved=bool(sc["success"]), detail=sc,
                       steps=[dict(i=0, tool="start", q=[round(float(x), 4) for x in q0], score=None),
                              dict(i=1, tool="final", q=[round(float(x), 4) for x in s.q()], score=round(100 * sc["hasta_score"], 2))],
                       wall_s=round(time.time() - t0, 1), **info)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(n, rec["score"], rec["solved"], rec["wall_s"], flush=True)


if __name__ == "__main__":
    main()
