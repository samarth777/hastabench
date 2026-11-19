"""Package data + high-res stills for the HyperFrames video (run as a GPU job from the repo root)."""
import json, os, shutil
import numpy as np
from PIL import Image
from hastabench.core import HastaCore, load_targets
from hastabench.hastas import ALL28

OUT = "video/hasta/assets"
os.makedirs(f"{OUT}/img", exist_ok=True)


def load_js(path):
    s = open(path).read()
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


S = load_js("site/data/summary.js")
T1, T2 = load_targets(), load_targets("samyuta_v1")
shutil.copy("site/data/hand1.js", f"{OUT}/hand1.js")
shutil.copy("site/vendor/three.min.js", "video/hasta/vendor/three.min.js")

c1 = HastaCore(img=640)
c2 = HastaCore(hands=("right", "left"), img=640)


def still(core, q, name, cam="front"):
    core.sim.set_kinematic(np.array(q))
    core.sim.command(np.array(q), steps=1)
    Image.fromarray(core.sim.render(cam, size=640)).save(f"{OUT}/img/{name}.jpg", quality=92)


# mosaic of all 28 targets
for t in ALL28:
    Image.open(os.path.join(T1[t]["_dir"], T1[t]["images"]["front"])).convert("RGB").resize((256, 256)).save(f"{OUT}/img/m_{t}.jpg", quality=88)


def ep(model_key, t):
    return load_js(f"site/data/ep/{model_key}.js")[t]


key = {m["model"]: m["key"] for m in S["models"] + S["two_models"] if True}
name = {m["key"]: m["name"] for m in S["models"] + S["two_models"]}

# showcase episode: Gemini 3.8 Flash on Mayura
gk = [m["key"] for m in S["models"] if m["model"] == "gcp/gemini-3.8-flash"][0]
E = ep(gk, "mayura")
episode = dict(model=name[gk], target="mayura", meaning=T1["mayura"]["meaning"], score=E["score"], solved=E["solved"],
               steps=[dict(t=s["t"], a=s["a"], ok=s["ok"], s=s["s"], q=s["q"]) for s in E["steps"]],
               plan=E["plan"][:240])
still(c1, T1["mayura"]["q"], "t_mayura")

# failures (target vs final pose)
fails = []
for model, t, note in [("claude-sonnet-5-5@high", "bhramara", "turned the hand ~170° away"),
                       ("gcp/gemini-3.8-flash", "katakamukha", "thumb misses both fingertips"),
                       ("claude-opus-5-5", "mayura", "reached 89, then walked away to 64"),
                       ("azure/gpt-5.6-sol", "hamsasya", "the pinch never closes")]:
    k = [m["key"] for m in S["models"] if m["model"] == model][0]
    e = ep(k, t)
    still(c1, T1[t]["q"], f"t_{t}")
    still(c1, e["steps"][-1]["q"], f"f_{k}_{t}")
    fails.append(dict(model=name[k], target=t, meaning=T1[t]["meaning"], score=round(e["score"], 1),
                      best=round(e.get("best", e["score"]), 1), note=note, timg=f"t_{t}", fimg=f"f_{k}_{t}"))

# two hands
two = []
for model, t in [("claude-opus-5-5", "anjali"), ("claude-opus-5-5", "swastika")]:
    k = [m["key"] for m in S["two_models"] if m["model"] == model][0]
    e = ep(k, t)
    still(c2, T2[t]["q"], f"t2_{t}")
    still(c2, e["steps"][-1]["q"], f"f2_{k}_{t}")
    two.append(dict(model=name[k], target=t, meaning=T2[t]["meaning"], score=round(e["score"], 1), timg=f"t2_{t}", fimg=f"f2_{k}_{t}"))

morph = [dict(name="rest", q=[0.0] * len(T1["pataka"]["q"]))] + [dict(name=t, meaning=T1[t]["meaning"], q=T1[t]["q"]) for t in
                                                                   ["pataka", "mayura", "shikhara", "alapadma"]]
board = [dict(name=m["name"], model=m["model"], mean=round(m["mean"], 1), solved=m["solved"]) for m in S["models"]]
refs = {m["model"]: round(m["mean"], 1) for m in S["refs"]}
data = dict(episode=episode, fails=fails, two=two, morph=morph, board=board, refs=refs, all28=ALL28,
            meanings={t: T1[t]["meaning"] for t in ALL28},
            n_eps=sum(m["n"] for m in S["models"] + S["two_models"]), n_models=len(S["models"]))
with open(f"{OUT}/data.js", "w") as f:
    f.write("window.HV = " + json.dumps(data, separators=(",", ":")) + ";\n")
print("ok", len(episode["steps"]), "steps;", [f["target"] for f in fails], data["n_eps"])
