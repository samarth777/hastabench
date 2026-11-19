"""Export everything the HTML report needs into site/data and site/img.

Data is written as JS files that assign globals, so index.html works from file://.
"""
import glob, json, os, re
from collections import defaultdict, Counter

import numpy as np
import mujoco
from PIL import Image

from hastabench.core import HastaCore, load_targets
from hastabench.hastas import ALL28, CORE8, DESCRIPTIONS, HASTAS, SAMYUTA
from hastabench.bench import Episode, BUDGET
from hastabench.scorer import KP_TOL_MM, FINGER_TOL_MM, ORIENT_TOL_DEG

SITE = "site"
os.makedirs(f"{SITE}/data/ep", exist_ok=True)
os.makedirs(f"{SITE}/img/final", exist_ok=True)
os.makedirs(f"{SITE}/img/target", exist_ok=True)

PRICE = {  # $/token from the LiteLLM proxy /model/info (input, cached input, output)
    "claude-opus-5-5": (3.04e-06, 1.52e-07, 1.52e-05), "claude-sonnet-5-5": (1.52e-06, 1.52e-07, 7.6e-06),
    "claude-haiku-4-5": (7.6e-07, 1.1e-07, 3.8e-06), "azure/gpt-5.6-sol": (5e-06, 5e-07, 3e-05),
    "azure/gpt-5.6-luna-ncf": (2e-07, 2e-08, 1.2e-06), "Azure/gpt-5.2-2025-12-11": (1.75e-06, 1.75e-07, 1.4e-05),
    "azure/gpt-5-mini": (2.5e-07, 2.5e-08, 2e-06), "gcp/gemini-3.8-flash": (7.5e-07, 7.5e-08, 3.75e-06),
    "gcp/gemini-2.5-pro": (1.25e-06, 1.25e-07, 1e-05), "azure/Kimi-K2.5": (6e-07, 6e-07, 3e-06)}
NICE = {"claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5", "claude-haiku-4-5": "Claude Haiku 4.5",
        "azure/gpt-5.6-sol": "GPT-5.6 Sol", "azure/gpt-5.6-luna-ncf": "GPT-5.6 Luna", "Azure/gpt-5.2-2025-12-11": "GPT-5.2",
        "azure/gpt-5-mini": "GPT-5 mini", "gcp/gemini-3.8-flash": "Gemini 3.8 Flash", "gcp/gemini-2.5-pro": "Gemini 2.5 Pro",
        "azure/Kimi-K2.5": "Kimi K2.5",
        "REF_rest": "Hands at rest", "REF_mean_shape": "Average-hasta prior", "REF_cma_image": "CMA-ES on silhouettes",
        "REF_cma_score": "Blind CMA-ES on hidden score", "REF_ik_oracle": "IK oracle (privileged)"}


def js(name, obj, path):
    with open(path, "w") as f:
        f.write(f"window.{name} = " + json.dumps(obj, separators=(",", ":")) + ";\n")


def r4(a):
    return [round(float(x), 5) for x in np.ravel(a)]


# ------------------------------------------------------------------ hand models
def export_model(core, key):
    s = core.sim
    m = s.m
    bodies = []
    for b in range(1, m.nbody):
        joints = []
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            nm = m.joint(j).name
            joints.append(dict(type="slide" if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE else "hinge",
                               axis=r4(m.jnt_axis[j]), q=s.joint_names.index(nm)))
        bodies.append(dict(name=m.body(b).name, parent=int(m.body_parentid[b]) - 1, pos=r4(m.body_pos[b]), quat=r4(m.body_quat[b]), joints=joints))
    skins = []
    for k in range(m.nskin):
        va, vn = m.skin_vertadr[k], m.skin_vertnum[k]
        fa, fn = m.skin_faceadr[k], m.skin_facenum[k]
        bones = []
        for bi in range(m.skin_boneadr[k], m.skin_boneadr[k] + m.skin_bonenum[k]):
            a, n = m.skin_bonevertadr[bi], m.skin_bonevertnum[bi]
            bones.append(dict(body=int(m.skin_bonebodyid[bi]) - 1, v=[int(x) for x in m.skin_bonevertid[a:a + n]],
                              w=[round(float(x), 4) for x in m.skin_bonevertweight[a:a + n]]))
        skins.append(dict(vert=r4(m.skin_vert[va:va + vn]), face=[int(x) for x in m.skin_face[fa:fa + fn].ravel()], bones=bones))
    meshes = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[g]
            va, vn = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
            fa, fn = m.mesh_faceadr[mi], m.mesh_facenum[mi]
            meshes.append(dict(body=int(m.geom_bodyid[g]) - 1, pos=r4(m.geom_pos[g]), quat=r4(m.geom_quat[g]),
                               vert=r4(m.mesh_vert[va:va + vn]), face=[int(x) for x in m.mesh_face[fa:fa + fn].ravel()]))
    cams = {}
    for c in range(m.ncam):
        cams[m.camera(c).name] = dict(pos=r4(m.cam_pos[c]), quat=r4(m.cam_quat[c]), fovy=float(m.cam_fovy[c]))
    return dict(key=key, nq=s.n, joint_names=s.joint_names, bodies=bodies, skins=skins, meshes=meshes, cams=cams,
                skin_rgb=[0.87, 0.68, 0.56])


core1 = HastaCore(img=384)
core2 = HastaCore(hands=("right", "left"), img=384)
js("HB_HAND1", export_model(core1, "one"), f"{SITE}/data/hand1.js")
js("HB_HAND2", export_model(core2, "two"), f"{SITE}/data/hand2.js")

# ------------------------------------------------------------------ targets
T1, T2 = load_targets(), load_targets("samyuta_v1")
T2 = {k: v for k, v in T2.items() if v["valid"]}
targets = {}
for tid, t in list(T1.items()) + list(T2.items()):
    two = tid in T2
    for v in ("front", "three_quarter", "side", "top"):
        Image.open(os.path.join(t["_dir"], t["images"][v])).convert("RGB").resize((256, 256)).save(f"{SITE}/img/target/{tid}_{v}.jpg", quality=88)
    info = dict(id=tid, two=two, meaning=t["meaning"], desc=DESCRIPTIONS.get(tid, ""), q=r4(t["q"]), core8=tid in CORE8)
    if two:
        info["shapes"] = [t["hands"]["R_"]["shape"], t["hands"]["L_"]["shape"]]
        info["contacts"] = t["cross_contacts"]
    else:
        info["contacts"] = t["contacts"]
        info["apart"] = t["apart"]
    targets[tid] = info

# example look composite (start pose of pataka, seed 0)
ep = Episode(core1, T1["mayura"], 0)
ep.composite("front").save(f"{SITE}/img/look_example.png")

# ------------------------------------------------------------------ episodes
rows = defaultdict(dict)        # model -> target -> record
for p in sorted(glob.glob("results/bench/*.jsonl")):
    for l in open(p):
        r = json.loads(l)
        if r.get("effort"):
            r["model_base"] = r["model"]
            r["model"] = r["model"] + "@" + r["effort"]
        rows[r["model"]][r["target"]] = r


def cost(r):
    pr = PRICE.get(r.get("model_base", r["model"]))
    if not pr or "usage" not in r:
        return None
    u = r["usage"]
    return (u["prompt"] - u.get("cached", 0)) * pr[0] + u.get("cached", 0) * pr[1] + u["completion"] * pr[2]


def boot_ci(x, n=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def render_final(core, q, path):
    core.sim.set_kinematic(np.array(q))
    Image.fromarray(core.sim.render("front", size=200)).save(path, quality=85)


def aliases(tid):
    a = {tid, tid.replace("sh", "s"), tid[:-1] if tid.endswith("a") else tid}
    return a


summary = dict(models=[], refs=[], two_models=[], two_refs=[])
curves, quality, refusals, failure, knowledge = {}, {}, {}, {}, {}
for model, eps in rows.items():
    is_ref = model.startswith("REF_")
    for split, tset in (("one", T1), ("two", T2)):
        E = {k: v for k, v in eps.items() if k in tset}
        if not E:
            continue
        key = re.sub(r"[^A-Za-z0-9]+", "_", model) + "_" + split
        core = core1 if split == "one" else core2
        epdata = {}
        for tid, r in E.items():
            final_q = r["steps"][-1]["q"]
            render_final(core, final_q, f"{SITE}/img/final/{key}_{tid}.jpg")
            if not is_ref:
                tr = [dict(turn=t["turn"], text=t.get("text", "")[:1500], step=t.get("step_from", 0), error=t.get("error")) for t in r["transcript"]]
                epdata[tid] = dict(score=r["score"], raw=r["raw_score"], solved=r["solved"], best=r["best_score_in_episode"], plan=r["plan"][:2000],
                                   reads=r["reads"], looks=r["looks"], calls=r["calls"], cost=cost(r), wall=r["wall_s"],
                                   steps=[dict(t=s["tool"], a=json.dumps(s["args"])[:400], ok=s["accepted"], m=s["msg"][:220], s=s["score"], q=s["q"]) for s in r["steps"]],
                                   transcript=tr)
            else:
                epdata[tid] = dict(score=r["score"], raw=r["raw_score"], solved=r["solved"], steps=[dict(t=s["tool"], a="", ok=True, m="", s=s["score"], q=s["q"]) for s in r["steps"]])
        js(f"HB_EP_{key}", epdata, f"{SITE}/data/ep/{key}.js")
        sc = [E[t]["score"] for t in E]
        base, _, eff = model.partition("@")
        ent = dict(key=key, model=model, base=base, effort=eff or "default", name=NICE.get(base, base) + (f" ({eff})" if eff else ""), split=split, n=len(E), mean=float(np.mean(sc)), ci=boot_ci(sc),
                   solved=int(sum(E[t]["solved"] for t in E)), per_target={t: E[t]["score"] for t in E},
                   solved_targets=[t for t in E if E[t]["solved"]])
        if not is_ref:
            cs = [cost(E[t]) for t in E]
            ent.update(cost=float(np.mean([c for c in cs if c is not None])) if any(c is not None for c in cs) else None,
                       time_min=float(np.mean([E[t]["wall_s"] for t in E]) / 60), calls=float(np.mean([E[t]["calls"] for t in E])),
                       looks=float(np.mean([E[t]["looks"] for t in E])), reads=float(np.mean([E[t]["reads"] for t in E])),
                       best=float(np.mean([E[t]["best_score_in_episode"] for t in E])),
                       lost=float(np.mean([E[t]["best_score_in_episode"] - E[t]["raw_score"] for t in E])),
                       lost_gt2=int(sum(E[t]["best_score_in_episode"] - E[t]["raw_score"] > 2 for t in E)),
                       tok_in=float(np.mean([E[t]["usage"]["prompt"] for t in E])), tok_out=float(np.mean([E[t]["usage"]["completion"] for t in E])))
            # per-call curves (score after call k; carried forward after submit)
            cur = np.zeros((len(E), BUDGET + 1)); best = np.zeros_like(cur)
            for i, r in enumerate(E.values()):
                v = [s["score"] for s in r["steps"]][: BUDGET + 1]
                v = v + [v[-1]] * (BUDGET + 1 - len(v))
                cur[i] = v; best[i] = np.maximum.accumulate(v)
            curves[key] = dict(cur=r4(cur.mean(0)), best=r4(best.mean(0)))
            # move quality + refusals
            q = Counter(); rf = Counter()
            for r in E.values():
                st = r["steps"]
                for a, b in zip(st[:-1], st[1:]):
                    if b["tool"] in ("set_fingers", "turn_hand", "move_hand", "undo"):
                        if not b["accepted"]:
                            m = b["msg"]
                            rf["per-call limit" if "per-call limit" in m else "joint range" if "range" in m else
                               "plan first" if "plan()" in m else "unknown joint / bad args"] += 1
                        else:
                            d = b["score"] - a["score"]
                            q["helped" if d > 0.5 else "hurt" if d < -0.5 else "neutral"] += 1
                tc = Counter(s["tool"] for s in st)
                for k in ("look", "describe", "read_description", "undo", "plan"):
                    q["n_" + k] += tc.get(k, 0)
            quality[key] = dict(q); refusals[key] = dict(rf)
            # failure modes (unsolved episodes)
            fm = Counter(); nf = 0
            for r in E.values():
                if r["solved"]:
                    continue
                nf += 1
                d = r["detail"]
                if split == "one":
                    fm["orientation"] += d["orient_err_deg"] > ORIENT_TOL_DEG
                    fm["shape"] += d["kp_err_mm"] > KP_TOL_MM or max(d["finger_err_mm"].values()) > FINGER_TOL_MM
                    fm["contacts"] += d["constraint_frac"] < 1
                    for f, v in d["finger_err_mm"].items():
                        fm["finger_" + f] += v > FINGER_TOL_MM
                else:
                    fm["relative pose"] += d["rel_rot_err_deg"] > ORIENT_TOL_DEG or d["rel_pos_err_mm"] > 25
                    fm["hand shapes"] += not (d["R_"]["success"] and d["L_"]["success"])
                    fm["cross contacts"] += not all(d["cross_contacts_ok"])
            failure[key] = dict(fm, n_fail=nf)
            # knowledge: did the plan name the right hasta?
            named = sum(any(a in r["plan"].lower() for a in aliases(t)) for t, r in E.items())
            readers = [r["score"] for r in E.values() if r["reads"] > 0]
            nonread = [r["score"] for r in E.values() if r["reads"] == 0]
            knowledge[key] = dict(named=int(named), n=len(E), read_eps=len(readers),
                                  score_read=float(np.mean(readers)) if readers else None,
                                  score_noread=float(np.mean(nonread)) if nonread else None)
        bucket = ("refs" if is_ref else "models") if split == "one" else ("two_refs" if is_ref else "two_models")
        summary[bucket].append(ent)

for k in summary:
    summary[k].sort(key=lambda e: -e["mean"])
# difficulty order (hardest first) from VLM mean score
diff = {t: np.mean([m["per_target"].get(t, np.nan) for m in summary["models"] if t in m["per_target"]]) for t in ALL28}
summary["order"] = sorted(ALL28, key=lambda t: diff[t] if not np.isnan(diff[t]) else 1e9)
summary["order_two"] = list(T2)
summary["all28"] = ALL28
summary.update(curves=curves, quality=quality, refusals=refusals, failure=failure, knowledge=knowledge, targets=targets,
               budget=BUDGET, scorer=json.load(open("results/scorer_validation.json")) if os.path.exists("results/scorer_validation.json") else None)
js("HB", summary, f"{SITE}/data/summary.js")
print("models", [(m["name"], round(m["mean"], 1), m["n"]) for m in summary["models"]])
print("refs", [(m["name"], round(m["mean"], 1), m["n"]) for m in summary["refs"]])
print("two", [(m["name"], round(m["mean"], 1), m["n"]) for m in summary["two_models"] + summary["two_refs"]])
