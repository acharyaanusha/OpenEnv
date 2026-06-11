# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

try:
    from openenv.core.env_server import Environment
    from openenv.core.env_server.types import State
except ImportError:  # standalone import path
    from core.env_server import Environment
    from core.env_server.types import State

from sophistry_bench_sprint import (
    alternation_canary,
    citation_grounding,
    claim_count_cliff,
    length_band_canary,
    load_quality_from_json,
    packaged_quality_path,
    parse_citations,
    parse_claims,
    quality_to_advocacy_dataset,
    starts_with_canary,
    template_echo_canary,
)

try:
    from ..models import AdvocacyAction, AdvocacyObservation
except ImportError:  # when imported as top-level package
    from sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation


_DEFAULT_WEIGHTS = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _weights_from_env() -> list[float]:
    raw = os.getenv("SPRINT_WEIGHTS")
    if not raw:
        return list(_DEFAULT_WEIGHTS)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    weights = [float(p) for p in parts]
    if len(weights) != 8:
        raise ValueError(f"SPRINT_WEIGHTS must have 8 values, got {len(weights)}")
    return weights


class SophistryBenchSprintEnvironment(Environment):
    """Single-step QuALITY advocacy environment (OpenEnv port).

    The dataset is built once at construction from the bundled QuALITY split,
    reusing the canonical sprint dataset builder. ``reset`` selects one advocacy
    row deterministically; ``step`` scores the completion with the canonical
    reward functions.
    """

    def __init__(
        self,
        n_items: Optional[int] = None,
        passage_chars: Optional[int] = None,
        seed: Optional[int] = None,
        weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.n_items = n_items if n_items is not None else int(os.getenv("SPRINT_N_ITEMS", "50"))
        self.passage_chars = (
            passage_chars
            if passage_chars is not None
            else int(os.getenv("SPRINT_PASSAGE_CHARS", "2000"))
        )
        self.build_seed = seed if seed is not None else int(os.getenv("SPRINT_SEED", "0"))
        self.weights = weights if weights is not None else _weights_from_env()

        items = load_quality_from_json(packaged_quality_path())
        if len(items) > self.n_items:
            items = items[: self.n_items]
        # HuggingFace Dataset of rows: {prompt, answer, info{passage,assigned_answer,is_gold,article_id}}
        self.dataset = quality_to_advocacy_dataset(
            items, seed=self.build_seed, passage_chars=self.passage_chars
        )
        self._n = len(self.dataset)
        if self._n == 0:
            raise RuntimeError("sprint dataset is empty; check bundled quality_dev.json")

        self._cursor = 0
        self._current_passage: str = ""
        self._current_is_gold: bool = False
        self._has_task = False
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

    @staticmethod
    def _system_text(row_prompt: list[dict]) -> str:
        for msg in row_prompt:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return row_prompt[0].get("content", "") if row_prompt else ""

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> AdvocacyObservation:
        if seed is not None:
            idx = seed % self._n
        else:
            idx = self._cursor
            self._cursor = (self._cursor + 1) % self._n

        row = self.dataset[idx]
        info = row["info"]
        self._current_passage = info["passage"]
        self._current_is_gold = bool(info["is_gold"])
        self._has_task = True

        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

        return AdvocacyObservation(
            prompt=self._system_text(row["prompt"]),
            answer_to_defend=info["assigned_answer"],
            item_id=info["article_id"],
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: AdvocacyAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AdvocacyObservation:
        # Scoring is implemented in the next task. This stub keeps the class
        # concrete (the base ``Environment.step`` is abstract) so ``reset`` can
        # be exercised on its own.
        raise NotImplementedError("step() scoring is implemented in the next task")

    @property
    def state(self) -> State:
        return self._state
