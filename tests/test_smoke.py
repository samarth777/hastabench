"""Fast smoke tests: model, targets, scorer and benchmark rules (~30 s, needs an EGL/OSMesa GL)."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from hastabench.core import HastaCore, load_targets
from hastabench.hastas import ALL28, CORE8
from hastabench.scorer import score_sim
from hastabench.bench import Episode, JOINT_CAP


@pytest.fixture(scope="module")
def core():
    return HastaCore()


@pytest.fixture(scope="module")
def targets():
    return load_targets()


def test_model_builds_without_penetration(core):
    assert core.sim.n == 27  # 6 root + 21 finger DOF
    core.sim.reset()
    core.sim.command(core.sim.q())
    assert core.sim.penetration() < 1e-3


def test_two_hand_model_builds():
    c = HastaCore(hands=("right", "left"))
    assert c.sim.n == 54


def test_all_targets_present_and_valid(targets):
    assert set(ALL28) <= set(targets) and set(CORE8) <= set(ALL28)
    assert all(targets[t]["valid"] for t in ALL28)
    two = load_targets("samyuta_v1")
    assert len(two) == 5 and all(v["valid"] for v in two.values())


@pytest.mark.parametrize("name", ["pataka", "mayura", "mushti", "trishula"])
def test_scorer_accepts_target_and_rejects_rest(core, targets, name):
    t = targets[name]
    core.sim.set_kinematic(np.array(t["q_command"]))
    core.sim.command(np.array(t["q_command"]), steps=400)
    assert score_sim(core.sim, t)["success"]
    core.reset(t, seed=0)
    assert not score_sim(core.sim, t)["success"]


def test_bench_rules(core, targets):
    ep = Episode(core, targets["pataka"], seed=0)
    out, _ = ep.call("set_fingers", {"hand": "right", "angles": {"index_mcp": 10}})
    assert out.startswith("REFUSED") and "plan()" in out          # plan first
    ep.call("plan", {"text": "pataka"})
    cur = core.sim.target()[core.sim.joint_names.index("index_mcp")] * 180 / np.pi
    out, _ = ep.call("set_fingers", {"hand": "right", "angles": {"index_mcp": cur + JOINT_CAP + 10}})
    assert "per-call limit" in out or "range" in out                # per-call cap / range
    q_before = core.sim.q().copy()
    out, _ = ep.call("set_fingers", {"hand": "right", "angles": {"index_mcp": cur + 20}})
    assert out.startswith("Accepted")
    ep.call("undo", {})
    assert np.allclose(core.sim.q(), q_before, atol=2e-2)            # undo restores
    _, img = ep.call("look", {"camera": "front"})
    assert img is not None and img.size[0] == 3 * img.size[1]        # target | current | diff
    ep.call("read_description", {})
    ep.call("submit", {})
    f = ep.final()
    assert f["reads"] == 1 and abs(f["score"] - 0.94 * f["raw_score"]) < 0.05
    assert all("q" in s and "score" in s for s in ep.steps)          # replayable log
