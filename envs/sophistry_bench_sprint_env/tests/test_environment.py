# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
"""Tests for the sophistry-bench sprint OpenEnv environment."""

from sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation


def test_advocacy_action_carries_text():
    a = AdvocacyAction(text="<claim>x</claim>")
    assert a.text == "<claim>x</claim>"


def test_advocacy_observation_defaults():
    o = AdvocacyObservation(prompt="P", answer_to_defend="A", item_id="id1")
    assert o.prompt == "P"
    assert o.answer_to_defend == "A"
    assert o.item_id == "id1"
    assert o.reward == 0.0
    assert o.done is False
    assert o.metadata == {}


def test_client_parses_step_result():
    from sophistry_bench_sprint_env.client import SophistryBenchSprintEnv
    from sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation

    # Exercise the pure parsing hooks without a live server.
    client = SophistryBenchSprintEnv.__new__(SophistryBenchSprintEnv)
    payload = client._step_payload(AdvocacyAction(text="<claim>x</claim>"))
    assert payload["text"] == "<claim>x</claim>"

    raw = {
        "observation": {
            "prompt": "",
            "answer_to_defend": "",
            "item_id": "",
            "reward": 0.5,
            "done": True,
            "metadata": {"aggregate_reward": 0.5},
        },
        "reward": 0.5,
        "done": True,
        "info": {},
    }
    result = client._parse_result(raw)
    assert isinstance(result.observation, AdvocacyObservation)
    assert result.observation.metadata["aggregate_reward"] == 0.5
    assert result.reward == 0.5
    assert result.done is True


from sophistry_bench_sprint_env.server.sophistry_bench_sprint_environment import (
    SophistryBenchSprintEnvironment,
)


def _env():
    # Small dataset keeps the test fast; reads the bundled QuALITY split.
    return SophistryBenchSprintEnvironment(n_items=2, passage_chars=500, seed=0)


def test_reset_returns_task_observation():
    env = _env()
    obs = env.reset(seed=0)
    assert obs.done is False
    assert obs.reward == 0.0
    assert obs.prompt  # non-empty system prompt
    assert "DEFEND THIS ANSWER" in obs.prompt
    assert obs.answer_to_defend in obs.prompt
    assert obs.item_id  # article id present


def test_reset_is_deterministic_for_fixed_seed():
    a = _env().reset(seed=3)
    b = _env().reset(seed=3)
    assert (a.item_id, a.answer_to_defend, a.prompt) == (
        b.item_id,
        b.answer_to_defend,
        b.prompt,
    )
