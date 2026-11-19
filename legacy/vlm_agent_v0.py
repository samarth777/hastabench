"""VLM benchmark track: a multimodal model poses the simulated hand via tool
calls, seeing only the target reference images and its own renders.

The hidden scorer is NEVER exposed: `evaluate()` just submits the final pose.

python -m hastabench.vlm_agent --model azure/gpt-5-mini --targets core8 --seeds 0 1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image
from openai import OpenAI

from hastabench.core import HastaCore, load_targets
from hastabench.hastas import CORE8, ALL28

FINGER_JOINTS_DOC = """Joints (degrees). Positive flexion always curls a finger toward the palm.
  thumb_cmc_flex [-30,70]  thumb base: sweeps the thumb across the palm (opposition)
  thumb_cmc_abd  [-45,60]  thumb base: + moves thumb away from the index (spread out), - presses it alongside the index
  thumb_cmc_rot  [-40,40]  thumb base axial twist
  thumb_mcp      [-15,65]  thumb middle knuckle flexion
  thumb_ip       [-20,85]  thumb tip joint flexion
  {f}_abd   [-20,20] (index/little [-25,25])  sideways spread; + = toward the thumb side
  {f}_mcp   [-25,95]  knuckle flexion (+ curls toward palm, - hyperextends)
  {f}_pip   [0,110]   middle joint flexion
  {f}_dip   [-5,85]   tip joint flexion
  where {f} is one of index, middle, ring, little."""

SYSTEM = f"""You control a simulated articulated RIGHT human hand in a 3-D physics simulator.
Your goal: reproduce the hand gesture (a classical Indian dance hasta/mudra) shown in the TARGET reference images
as exactly as possible: every finger's bend, spread, the thumb placement, which fingertips touch, and the overall
orientation of the hand as seen by the cameras.

Coordinate conventions: the front camera looks at the hand from the audience. With all joints at 0 the hand is
open and flat, fingers pointing straight up, palm facing the front camera, thumb on the image's right side.
{FINGER_JOINTS_DOC}
Whole-hand tools: rotate_wrist(roll, pitch, yaw) applies a RELATIVE rotation in degrees (roll = about the forearm axis,
pitch = tilting the fingers toward/away from the front camera, yaw = tilting the fingers left/right in the front view).
move_hand translates the hand (cm); position barely matters, shape and orientation do.

Cameras for render_view: front, side (from the hand's thumb side, i.e. image right), top, three_quarter, back.
Fingers collide physically: a finger pushed into another stops at the contact.
Each tool call costs one action from a fixed budget (shown in every reply). You may issue several tool calls per turn.
Call render_view to see your current hand. When you are satisfied, call evaluate() to submit (this ends the episode;
you get no score back)."""


SYSTEM2 = SYSTEM.replace("a simulated articulated RIGHT human hand", "TWO simulated articulated human hands (right and left)").replace(
    "With all joints at 0 the hand is\nopen and flat, fingers pointing straight up, palm facing the front camera, thumb on the image's right side.",
    "With all joints at 0 each hand is\nopen and flat, fingers up, palm facing the front camera; the right hand is on the image's LEFT (thumb toward the\nimage centre), the left hand on the image's right, as for a dancer facing the audience. Joint semantics are mirror-symmetric:\nthe same joint values give mirror-image poses on the two hands; move_hand dx>0 moves a hand toward the midline.").replace(
    "Your goal: reproduce the hand gesture", "Your goal: reproduce the two-hand gesture").replace(
    "the overall\norientation of the hand", "the overall\norientation of each hand, how the hands are placed relative to each other and where they touch")


def tool_schema(two=False):
    def fn(name, desc, props, req):
        return {"type": "function", "function": {"name": name, "description": desc,
                "parameters": {"type": "object", "properties": props, "required": req}}}
    hand = {"type": "string", "enum": ["right", "left"] if two else ["right"], "description": "which hand"}
    return [
        fn("set_joint", "Set one finger/thumb joint to an absolute angle in degrees.",
           {"hand": hand, "joint": {"type": "string", "description": "e.g. index_mcp, thumb_cmc_abd"},
            "angle": {"type": "number"}}, ["hand", "joint", "angle"]),
        fn("rotate_wrist", "Relative rotation of the whole hand in degrees.",
           {"hand": hand, "roll": {"type": "number"}, "pitch": {"type": "number"}, "yaw": {"type": "number"}}, ["hand"]),
        fn("move_hand", "Relative translation of the whole hand in cm.",
           {"hand": hand, "dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"}}, ["hand"]),
        fn("render_view", "Render the current hand from a camera; returns an image.",
           {"camera": {"type": "string", "enum": ["front", "side", "top", "three_quarter", "back"]}}, ["camera"]),
        fn("evaluate", "Submit the current pose as final. Ends the episode.", {}, []),
    ]


def img_url(arr_or_path, size=384):
    im = Image.open(arr_or_path) if isinstance(arr_or_path, str) else Image.fromarray(arr_or_path)
    im = im.convert("RGB").resize((size, size))
    b = io.BytesIO()
    im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def run_episode(client, model, core: HastaCore, target, seed, budget=40, max_turns=16, keep_images=3,
                target_views=("front", "three_quarter"), name_hint=False, log_fn=None):
    two = len(target.get("hands", [])) == 2 if isinstance(target.get("hands"), list) else True
    if two:
        target_views = tuple(target["views"])
    core.reset(target, seed)
    tdir = target["_dir"]
    content = [{"type": "text", "text": "TARGET reference images (" + ", ".join(f"{v} camera" for v in target_views) + "):"}]
    for v in target_views:
        content.append({"type": "image_url", "image_url": {"url": img_url(os.path.join(tdir, target["images"][v]))}})
    hint = f" The gesture is the hasta '{target['id']}' ({target['meaning']})." if name_hint else ""
    content.append({"type": "text", "text": f"Your hand currently (front camera) is shown next. The start pose is random.{hint} "
                    f"Action budget: {budget} tool calls."})
    content.append({"type": "image_url", "image_url": {"url": img_url(core.tool_render_view("front"))}})
    msgs = [{"role": "system", "content": SYSTEM2 if two else SYSTEM}, {"role": "user", "content": content}]
    tools = tool_schema(two)
    used, submitted, usage = 0, False, {"prompt": 0, "completion": 0}
    trace = []
    for turn in range(max_turns):
        # keep only the most recent `keep_images` renders (plus the targets) in context
        _prune_images(msgs, keep_images)
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(model=model, messages=msgs, tools=tools, tool_choice="auto")
                break
            except Exception as e:  # transient proxy errors
                err = str(e)
                time.sleep(5 * (attempt + 1))
        else:
            trace.append({"turn": turn, "error": err[:300]})
            break
        if resp.usage:
            usage["prompt"] += resp.usage.prompt_tokens or 0
            usage["completion"] += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        am = {"role": "assistant", "content": msg.content or ""}
        if calls:
            am["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]
        msgs.append(am)
        if not calls:
            trace.append({"turn": turn, "text": (msg.content or "")[:500], "calls": []})
            msgs.append({"role": "user", "content": "Please continue using the tools, or call evaluate() to submit."})
            if turn >= 2 and not any(t.get("calls") for t in trace[-2:]):
                break
            continue
        new_images = []
        tr = {"turn": turn, "text": (msg.content or "")[:500], "calls": []}
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "evaluate":
                submitted = True
                out = "Submitted."
            elif used >= budget:
                out = "Budget exhausted; call evaluate()."
            else:
                used += 1
                try:
                    if name == "set_joint":
                        out = core.tool_set_joint(args.get("hand", "right"), args["joint"], float(args["angle"]))
                    elif name == "rotate_wrist":
                        out = core.tool_rotate_wrist(args.get("hand", "right"), float(args.get("roll", 0)), float(args.get("pitch", 0)), float(args.get("yaw", 0)))
                    elif name == "move_hand":
                        out = core.tool_move_hand(args.get("hand", "right"), float(args.get("dx", 0)), float(args.get("dy", 0)), float(args.get("dz", 0)))
                    elif name == "render_view":
                        cam = args.get("camera", "front")
                        new_images.append((cam, core.tool_render_view(cam)))
                        out = f"Rendered {cam} view (image follows)."
                    else:
                        out = f"Unknown tool {name}."
                except Exception as e:
                    out = f"Error: {e}"
            out += f" [actions used {used}/{budget}]"
            tr["calls"].append({"name": name, "args": args, "out": out[:200]})
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
        trace.append(tr)
        if new_images:
            cont = []
            for cam, im in new_images:
                cont += [{"type": "text", "text": f"Current hand, {cam} camera:"}, {"type": "image_url", "image_url": {"url": img_url(im)}}]
            msgs.append({"role": "user", "content": cont})
        if submitted or used >= budget + 2:
            break
    score = core.evaluate()
    rec = dict(model=model, target=target["id"], seed=seed, submitted=submitted, actions=used, turns=len(trace),
               usage=usage, score=score, final_q_deg=core.sim.to_human(core.sim.q()).round(1).tolist(), trace=trace)
    if log_fn:
        log_fn(rec)
    return rec


def _prune_images(msgs, keep):
    idx = [i for i, m in enumerate(msgs) if m["role"] == "user" and isinstance(m["content"], list)
           and any(c.get("type") == "image_url" for c in m["content"])]
    render_msgs = idx[1:]  # idx[0] holds the targets (+ start render) -> always kept
    for i in render_msgs[:-keep] if len(render_msgs) > keep else []:
        msgs[i]["content"] = [c if c.get("type") != "image_url" else {"type": "text", "text": "[older render omitted]"} for c in msgs[i]["content"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="core8")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--name-hint", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    client = OpenAI(base_url=os.environ["ANTHROPIC_BASE_URL"].rstrip("/") + "/v1", api_key=os.environ["ANTHROPIC_AUTH_TOKEN"], timeout=180)
    if args.targets == "samyuta":
        T = load_targets("samyuta_v1")
        names = [k for k, v in T.items() if v["valid"]]
    else:
        T = load_targets()
        names = CORE8 if args.targets == "core8" else ALL28 if args.targets == "all28" else args.targets.split(",")
    tag = args.model.replace("/", "_") + ("_hint" if args.name_hint else "")
    out = args.out or f"results/vlm/{tag}_{args.targets}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r["target"], r["seed"]))
    core = HastaCore(hands=("right", "left") if args.targets == "samyuta" else ("right",), budget=args.budget)
    with open(out, "a") as f:
        for s in args.seeds:
            for n in names:
                if (n, s) in done:
                    continue
                t0 = time.time()
                r = run_episode(client, args.model, core, T[n], s, budget=args.budget, name_hint=args.name_hint)
                r["wall_s"] = round(time.time() - t0, 1)
                f.write(json.dumps(r) + "\n")
                f.flush()
                sc = r["score"]
                print(f"{n} s{s} success={sc['success']} hs={sc['hasta_score']:.3f} kp={sc['kp_err_mm']:.1f}mm "
                      f"orient={sc.get('orient_err_deg', sc.get('rel_rot_err_deg', -1)):.1f} acts={r['actions']} tok={r['usage']}", flush=True)


if __name__ == "__main__":
    main()
