"""Qualitative grid: target vs. final pose of each VLM (seed 0, core-8 hastas)."""
import glob, json, os, re
import numpy as np
from PIL import Image, ImageDraw
from hastabench.core import HastaCore, load_targets
from hastabench.hastas import CORE8

T = load_targets()
core = HastaCore(img=192)
runs = {}
for p in sorted(glob.glob("results/vlm/*all28*.jsonl")) + sorted(glob.glob("results/vlm/*core8*.jsonl")):
    key = re.sub(r"_s\d+$", "", os.path.basename(p)[:-6]).replace("_all28", "").replace("_core8", "")
    for l in open(p):
        r = json.loads(l)
        if r["seed"] == 0 and r["target"] in CORE8:
            runs.setdefault(key, {})[r["target"]] = r
cols = ["target"] + list(runs)
S = 192
grid = Image.new("RGB", (S * len(cols), S * len(CORE8) + 24), (252, 252, 251))
d = ImageDraw.Draw(grid)
for j, c in enumerate(cols):
    d.text((j * S + 4, 4), c[:30], fill=(11, 11, 11))
for i, h in enumerate(CORE8):
    tim = Image.open(os.path.join(T[h]["_dir"], T[h]["images"]["front"])).resize((S, S))
    grid.paste(tim, (0, 24 + i * S))
    d.text((4, 24 + i * S + 4), h, fill=(11, 11, 11))
    for j, c in enumerate(cols[1:], 1):
        r = runs[c].get(h)
        if r is None:
            continue
        core.sim.set_kinematic(core.sim.from_human(r["final_q_deg"]))
        im = Image.fromarray(core.sim.render("front", size=S))
        dd = ImageDraw.Draw(im)
        sc = r["score"]
        dd.text((4, 4), ("OK " if sc["success"] else "") + f"{sc['hasta_score']:.2f}", fill=(0, 110, 0) if sc["success"] else (160, 30, 30))
        grid.paste(im, (j * S, 24 + i * S))
os.makedirs("results/fig", exist_ok=True)
grid.save("results/fig/qualitative_core8_seed0.png")
print("saved", cols)
