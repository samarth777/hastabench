"""HastaBench v1 agent harness (ShadowBench-style mechanics, adapted to hastas).

The model sees reference renders of the target hasta and poses a simulated hand
with tool calls. It never sees its score. Every call is logged together with
the hand state and a hidden dense score, which makes episodes replayable.

Tools: plan, set_fingers, turn_hand, move_hand, undo, describe, look,
read_description (paid peek: final score x0.94 per read), submit.
Rules: plan before the first move; per call, each joint may change by at most 45 deg,
the hand may turn by at most 45 deg per axis and move by at most 10 cm; out-of-range or
over-limit calls are REFUSED (they consume a call and leave the pose unchanged).
Collisions are physical: a finger pushed into another stops at the contact.
Budget: 60 calls, at most 20 looks, at most 2 description reads.

python -m hastabench.bench --model claude-sonnet-5-5 --targets all28 --seed 0
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI

from .core import HastaCore, load_targets
from .hastas import CORE8, ALL28, DESCRIPTIONS
from .model import finger_joint_names, DEG
from .scorer import score_sim, score_two

BUDGET, MAX_LOOKS, MAX_READS = 60, 20, 2
JOINT_CAP, TURN_CAP, MOVE_CAP = 45.0, 45.0, 10.0
PEEK_PENALTY = 0.94
CAMS = ("front", "side", "top", "three_quarter", "back")
PANEL = 320

JOINT_DOC = """Finger joints (degrees; positive flexion always curls toward the palm):
  thumb_cmc_flex [-30,70]  thumb base: + sweeps the thumb across the palm toward the little finger (opposition)
  thumb_cmc_abd  [-45,40]  thumb base: - presses the thumb alongside the index finger, + spreads it away
  thumb_cmc_rot  [-40,40]  thumb base axial twist
  thumb_mcp      [-15,65]  thumb middle knuckle      thumb_ip [-20,85]  thumb tip joint
  F_abd [-20,20] (index/little [-25,25])  sideways spread, + = toward the thumb side
  F_mcp [-25,95] knuckle    F_pip [0,110] middle joint    F_dip [-5,85] tip joint
  with F in index, middle, ring, little. All zeros = flat open hand, fingers slightly splayed."""

SYSTEM_1 = f"""You control a simulated articulated RIGHT human hand in a 3-D physics simulator. Reproduce the classical
Indian dance hand gesture (hasta / mudra) shown in the TARGET images as exactly as possible: every finger's bend
and spread, the thumb, which fingertips touch, and the hand's orientation.

Conventions: the front camera looks at the hand from the audience. With all joints at 0 the hand is open with fingers
pointing up, palm facing the front camera, thumb on the image's right.
{JOINT_DOC}
turn_hand rotates the whole hand RELATIVELY (degrees): roll = about the forearm axis, pitch = tip the fingers toward
(+) or away from the front camera, yaw = tilt the fingers toward the image right (+) / left in the front view.
move_hand translates the hand (cm); position barely matters, shape and orientation do.

Rules: call plan() before your first move. Per call each joint may change by at most {JOINT_CAP:.0f} deg, turns are at most
{TURN_CAP:.0f} deg per axis and moves at most {MOVE_CAP:.0f} cm per axis; a call that breaks a limit or a joint range is refused
(it still costs a call). Fingers collide physically. Budget: {BUDGET} calls in total, of which at most {MAX_LOOKS} looks
and {MAX_READS} description reads. look(camera) shows [target | your hand | difference] from that camera (difference:
red = target only, blue = your hand only). read_description() returns the classical text definition of the gesture
but multiplies your final score by {PEEK_PENALTY}. You never see your score. Call submit() when done."""

SYSTEM_2 = SYSTEM_1.replace("a simulated articulated RIGHT human hand", "TWO simulated articulated human hands (right and left)").replace(
    "Reproduce the classical\nIndian dance hand gesture", "Reproduce the classical\nIndian dance TWO-HAND gesture").replace(
    "and the hand's orientation.", "each hand's orientation, how the two hands are placed relative to each other and where they touch.").replace(
    "With all joints at 0 the hand is open with fingers\npointing up, palm facing the front camera, thumb on the image's right.",
    "With all joints at 0 each hand is open, fingers up, palm facing the front camera;\nthe right hand is on the image's LEFT (thumb toward the image centre) and the left hand on the right, like a dancer facing\nthe audience. Joints are mirror-symmetric (same values = mirror-image poses). move_hand dx>0 moves a hand toward the midline.").replace(
    "yaw = tilt the fingers toward the image right (+) / left in the front view.",
    "yaw = tilt the fingers toward the thumb side (+).")


def tools(two):
    hand = {"type": "string", "enum": ["right", "left"] if two else ["right"]}
    f = lambda n, d, p, r: {"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": r}}}
    num = {"type": "number"}
    return [
        f("plan", "State which gesture you think the target is and how each finger/thumb must be placed. Required before the first move; may be revised.", {"text": {"type": "string"}}, ["text"]),
        f("set_fingers", "Set one or more finger/thumb joints of one hand to absolute angles (degrees), e.g. {\"index_mcp\": 40, \"thumb_ip\": 20}.",
          {"hand": hand, "angles": {"type": "object", "additionalProperties": {"type": "number"}}}, ["hand", "angles"]),
        f("turn_hand", "Rotate a whole hand relatively (degrees).", {"hand": hand, "roll": num, "pitch": num, "yaw": num}, ["hand"]),
        f("move_hand", "Translate a whole hand relatively (cm).", {"hand": hand, "dx": num, "dy": num, "dz": num}, ["hand"]),
        f("undo", "Revert the last accepted change.", {}, []),
        f("describe", "Return the current joint angles, ranges and hand placement as numbers.", {}, []),
        f("look", "Return an image [target | your hand | difference] from one camera.", {"camera": {"type": "string", "enum": list(CAMS)}}, ["camera"]),
        f("read_description", "Return the classical text definition of the target gesture (final score x0.94 per read).", {}, []),
        f("submit", "Submit the current pose as final and end the episode.", {}, []),
    ]


def png_url(img: Image.Image):
    b = io.BytesIO()
    img.convert("RGB").save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def skin_mask(a):
    a = a.astype(np.int16)
    return ((a[..., 0] - a[..., 2]) > 25) & (a[..., 0] > 60)


class Episode:
    def __init__(self, core: HastaCore, target: dict, seed: int):
        self.core, self.t, self.seed = core, target, seed
        self.two = len(core.sim.prefixes) == 2
        core.reset(target, seed)
        self.planned = False
        self.plan_text = ""
        self.undo_stack = []
        self.calls = self.looks = self.reads = 0
        self.done = False
        self.steps = []
        self._timgs = {}
        self.log_step("start", {}, True, "")

    # ------------------------------------------------------------ helpers
    def hidden(self):
        sc = score_two(self.core.sim, self.t) if self.two else score_sim(self.core.sim, self.t)
        return sc

    def log_step(self, tool, args, accepted, msg):
        sc = self.hidden()
        self.steps.append(dict(i=len(self.steps), tool=tool, args=args, accepted=bool(accepted), msg=msg[:300],
                               score=round(100 * sc["hasta_score"], 2), success=bool(sc["success"]),
                               q=[round(float(x), 4) for x in self.core.sim.q()]))

    def pre(self, hand):
        hand = (hand or "right").lower()
        if not self.two:
            if hand != "right":
                raise ValueError("there is only a right hand")
            return ""
        if hand not in ("right", "left"):
            raise ValueError("hand must be right or left")
        return "R_" if hand == "right" else "L_"

    def target_img(self, cam):
        if cam not in self._timgs:
            self._timgs[cam] = np.array(Image.open(os.path.join(self.t["_dir"], self.t["images"][cam])).convert("RGB").resize((PANEL, PANEL)))
        return self._timgs[cam]

    def composite(self, cam):
        tgt = self.target_img(cam)
        cur = self.core.sim.render(cam, size=PANEL)
        mt, mc = skin_mask(tgt), skin_mask(cur)
        diff = np.full_like(tgt, 245)
        diff[mt & mc] = (190, 190, 190)
        diff[mt & ~mc] = (220, 40, 40)
        diff[~mt & mc] = (40, 90, 220)
        img = Image.fromarray(np.concatenate([tgt, cur, diff], 1))
        d = ImageDraw.Draw(img)
        for k, lab in enumerate([f"TARGET ({cam})", f"YOUR HAND ({cam})", "DIFFERENCE red=target only blue=yours only"]):
            d.rectangle([k * PANEL, 0, k * PANEL + 8 + 6.2 * len(lab), 16], fill=(255, 255, 255))
            d.text((k * PANEL + 4, 3), lab, fill=(0, 0, 0))
        return img

    def apply(self, deltas: dict):
        """deltas {joint: change in internal units}; returns settled q."""
        self.undo_stack.append((self.core.sim.q(), self.core.sim.target()))
        self.core.step_vector(self.core.delta_to_action(deltas))

    # ------------------------------------------------------------ tools
    def call(self, name, args):
        """Returns (text, image or None)."""
        if self.done:
            return "Episode already submitted.", None
        if self.calls >= BUDGET and name != "submit":
            return "Budget exhausted; the pose has been submitted.", None
        self.calls += 1
        s = self.core.sim
        img = None
        try:
            if name == "plan":
                self.planned = True
                self.plan_text = str(args.get("text", ""))[:2000]
                out, ok = "Plan recorded.", True
            elif name in ("set_fingers", "turn_hand", "move_hand") and not self.planned:
                out, ok = "REFUSED: call plan() before the first move.", False
            elif name == "set_fingers":
                p = self.pre(args.get("hand", "right"))
                angles = args.get("angles") or {}
                if not isinstance(angles, dict) or not angles:
                    raise ValueError("angles must be a non-empty object {joint: degrees}")
                allowed = set(finger_joint_names())
                tgt = s.target()
                deltas, probs = {}, []
                for j, v in angles.items():
                    if j not in allowed:
                        probs.append(f"unknown joint '{j}'")
                        continue
                    i = s.joint_names.index(p + j)
                    lo, hi = s.lo[i] / DEG, s.hi[i] / DEG
                    v = float(v)
                    if v < lo - 1e-6 or v > hi + 1e-6:
                        probs.append(f"{j}={v:g} outside range [{lo:.0f},{hi:.0f}]")
                    elif abs(v - tgt[i] / DEG) > JOINT_CAP + 1e-6:
                        probs.append(f"{j} change {v - tgt[i] / DEG:+.0f} deg exceeds the {JOINT_CAP:.0f} deg per-call limit")
                    else:
                        deltas[p + j] = v * DEG - tgt[i]
                if probs:
                    out, ok = "REFUSED: " + "; ".join(probs) + ". Pose unchanged.", False
                else:
                    self.apply(deltas)
                    q = s.q()
                    rep = []
                    for j in deltas:
                        i = s.joint_names.index(j)
                        want = (tgt[i] + deltas[j]) / DEG
                        rep.append(f"{j[len(p):]}={q[i] / DEG:.0f}" + ("(blocked by contact)" if abs(q[i] / DEG - want) > 4 else ""))
                    out, ok = "Accepted. Settled: " + ", ".join(rep), True
            elif name in ("turn_hand", "move_hand"):
                p = self.pre(args.get("hand", "right"))
                keys = ("roll", "pitch", "yaw") if name == "turn_hand" else ("dx", "dy", "dz")
                vals = [float(args.get(k, 0) or 0) for k in keys]
                cap = TURN_CAP if name == "turn_hand" else MOVE_CAP
                jn = [p + "root_" + k for k in (("roll", "pitch", "yaw") if name == "turn_hand" else ("tx", "ty", "tz"))]
                unit = DEG if name == "turn_hand" else 0.01
                tgt = s.target()
                probs = []
                for k, v, j in zip(keys, vals, jn):
                    i = s.joint_names.index(j)
                    if abs(v) > cap + 1e-6:
                        probs.append(f"{k}={v:g} exceeds the per-call limit {cap:.0f}")
                    elif not (s.lo[i] - 1e-6 <= tgt[i] + v * unit <= s.hi[i] + 1e-6):
                        probs.append(f"{k} would leave the range [{s.lo[i] / unit:.0f},{s.hi[i] / unit:.0f}]")
                if probs:
                    out, ok = "REFUSED: " + "; ".join(probs) + ". Pose unchanged.", False
                else:
                    self.apply({j: v * unit for j, v in zip(jn, vals)})
                    q = s.q()
                    cur = [q[s.joint_names.index(j)] / unit for j in jn]
                    lab = ("roll", "pitch", "yaw") if name == "turn_hand" else ("x", "y", "z")
                    out, ok = "Accepted. Now " + ", ".join(f"{l}={c:.1f}" for l, c in zip(lab, cur)) + (" deg" if name == "turn_hand" else " cm"), True
            elif name == "undo":
                if not self.undo_stack:
                    out, ok = "Nothing to undo.", False
                else:
                    q, tg = self.undo_stack.pop()
                    s.reset(q)
                    s.d.ctrl[s.act_ids] = tg
                    s.command(tg)
                    out, ok = "Reverted the last change.", True
            elif name == "describe":
                q = s.q()
                lines = []
                for p in s.prefixes:
                    hand = {"": "right", "R_": "right", "L_": "left"}[p]
                    fj = ", ".join(f"{j}={q[s.joint_names.index(p + j)] / DEG:.0f}" for j in finger_joint_names())
                    r = [q[s.joint_names.index(p + "root_" + k)] for k in ("tx", "ty", "tz", "roll", "pitch", "yaw")]
                    lines.append(f"{hand} hand: {fj}; offset x={r[0] * 100:.1f} y={r[1] * 100:.1f} z={r[2] * 100:.1f} cm; "
                                 f"roll={r[3] / DEG:.0f} pitch={r[4] / DEG:.0f} yaw={r[5] / DEG:.0f} deg")
                out, ok = "\n".join(lines), True
            elif name == "look":
                cam = args.get("camera", "front")
                if cam not in CAMS:
                    raise ValueError(f"camera must be one of {CAMS}")
                if self.looks >= MAX_LOOKS:
                    out, ok = "REFUSED: look limit reached.", False
                else:
                    self.looks += 1
                    img = self.composite(cam)
                    out, ok = f"Image follows ({cam}).", True
            elif name == "read_description":
                if self.reads >= MAX_READS:
                    out, ok = "REFUSED: description read limit reached.", False
                else:
                    self.reads += 1
                    out, ok = DESCRIPTIONS.get(self.t["id"], "(no description)"), True
            elif name == "submit":
                self.done = True
                out, ok = "Submitted.", True
            else:
                out, ok = f"Unknown tool {name}.", False
        except Exception as e:  # malformed arguments
            out, ok = f"REFUSED: {e}", False
        self.log_step(name, args, ok, out)
        left = BUDGET - self.calls
        return out + f" [calls left {left}, looks left {MAX_LOOKS - self.looks}]", img

    def final(self):
        sc = self.hidden()
        pen = PEEK_PENALTY ** self.reads
        best = max(st["score"] for st in self.steps)
        return dict(score=round(100 * sc["hasta_score"] * pen, 2), raw_score=round(100 * sc["hasta_score"], 2),
                    solved=bool(sc["success"]), best_score_in_episode=best, reads=self.reads, looks=self.looks,
                    calls=self.calls, detail=sc)


def run(client, model, core, target, seed, effort=None, max_turns=50, keep_images=3, log=print):
    ep = Episode(core, target, seed)
    two = ep.two
    views = ("front", "top") if two else ("front", "three_quarter")
    content = [{"type": "text", "text": "TARGET reference images (" + " and ".join(views) + " cameras):"}]
    for v in views:
        content.append({"type": "image_url", "image_url": {"url": png_url(Image.fromarray(ep.target_img(v)))}})
    content.append({"type": "text", "text": "Your hand(s) at the start (random pose), front camera:"})
    content.append({"type": "image_url", "image_url": {"url": png_url(Image.fromarray(core.sim.render("front", size=PANEL)))}})
    content.append({"type": "text", "text": "Begin with plan()."})
    msgs = [{"role": "system", "content": SYSTEM_2 if two else SYSTEM_1}, {"role": "user", "content": content}]
    T = tools(two)
    usage = {"prompt": 0, "completion": 0, "cached": 0}
    transcript = []
    t0 = time.time()
    extra = {}
    if effort:
        extra["reasoning_effort"] = effort
    idle = 0
    for turn in range(max_turns):
        _prune(msgs, keep_images)
        err = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(model=model, messages=msgs, tools=T, tool_choice="auto", **extra)
                err = None
                break
            except Exception as e:
                err = str(e)
                time.sleep(8 * (attempt + 1))
        if err:
            transcript.append({"turn": turn, "error": err[:400]})
            break
        u = resp.usage
        if u:
            usage["prompt"] += u.prompt_tokens or 0
            usage["completion"] += u.completion_tokens or 0
            d = getattr(u, "prompt_tokens_details", None)
            usage["cached"] += (getattr(d, "cached_tokens", 0) or 0) if d else 0
        m = resp.choices[0].message
        calls = m.tool_calls or []
        am = {"role": "assistant", "content": m.content or ""}
        if calls:
            am["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]
        msgs.append(am)
        transcript.append({"turn": turn, "text": (m.content or "")[:3000], "step_from": len(ep.steps)})
        if not calls:
            idle += 1
            if idle >= 3:
                break
            msgs.append({"role": "user", "content": "Continue with tool calls, or call submit()."})
            continue
        idle = 0
        imgs = []
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            out, img = ep.call(c.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            if img is not None:
                imgs.append(img)
        if imgs:
            cont = []
            for im in imgs:
                cont += [{"type": "text", "text": "look() result:"}, {"type": "image_url", "image_url": {"url": png_url(im)}}]
            msgs.append({"role": "user", "content": cont})
        if ep.done or ep.calls >= BUDGET:
            break
    if not ep.done:
        ep.call("submit", {})
    res = ep.final()
    rec = dict(model=model, effort=effort, target=target["id"], seed=seed, two_hand=two, **res, plan=ep.plan_text,
               usage=usage, wall_s=round(time.time() - t0, 1), steps=ep.steps, transcript=transcript)
    log(f"{target['id']} s{seed} score={res['score']:.1f} solved={res['solved']} best={res['best_score_in_episode']:.1f} "
        f"calls={res['calls']} looks={res['looks']} reads={res['reads']} tok={usage} {rec['wall_s']}s")
    return rec


def _prune(msgs, keep):
    idx = [i for i, m in enumerate(msgs) if m["role"] == "user" and isinstance(m["content"], list)
           and any(c.get("type") == "image_url" for c in m["content"])]
    for i in idx[1:-keep] if len(idx) > keep + 1 else []:
        msgs[i]["content"] = [c if c.get("type") != "image_url" else {"type": "text", "text": "[older image omitted]"} for c in msgs[i]["content"]]


def make_client(timeout=600):
    """OpenAI-compatible client. Uses OPENAI_BASE_URL / OPENAI_API_KEY; falls back to a
    LiteLLM-style proxy given by ANTHROPIC_BASE_URL (+ '/v1') / ANTHROPIC_AUTH_TOKEN."""
    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    if not base and os.environ.get("ANTHROPIC_BASE_URL"):
        base = os.environ["ANTHROPIC_BASE_URL"].rstrip("/") + "/v1"
        key = key or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    return OpenAI(base_url=base, api_key=key, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="all28")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    client = make_client()
    if args.targets == "samyuta":
        T = load_targets("samyuta_v1")
        names = [k for k, v in T.items() if v["valid"]]
        core = HastaCore(hands=("right", "left"))
    else:
        T = load_targets()
        names = CORE8 if args.targets == "core8" else ALL28 if args.targets == "all28" else args.targets.split(",")
        core = HastaCore()
    tag = args.model.replace("/", "_") + (f"_{args.effort}" if args.effort else "")
    out = args.out or f"results/bench/{tag}_{args.targets}_s{args.seed}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["target"] for l in open(out)} if os.path.exists(out) else set()
    with open(out, "a") as f:
        for n in names:
            if n in done:
                continue
            r = run(client, args.model, core, T[n], args.seed, args.effort, log=lambda s: print(s, flush=True))
            f.write(json.dumps(r) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
