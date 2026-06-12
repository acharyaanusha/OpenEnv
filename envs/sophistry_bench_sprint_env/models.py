# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:  # standalone import path
    from core.env_server.types import Action, Observation

from pydantic import Field


class AdvocacyAction(Action):
    """The policy's one-shot advocacy argument."""

    text: str = Field(
        ..., description="The argument completion, using <claim>/<cite> tags."
    )


class AdvocacyObservation(Observation):
    """Task on reset; scored result on step.

    On reset: ``prompt`` holds the full system prompt (passage + question +
    answer-to-defend), ``done`` is False.
    On step: ``prompt`` is empty, ``done`` is True, and ``metadata`` carries all
    eight reward components.

    Note on ``reward``: read the post-step reward from ``StepResult.reward``, not
    from ``observation.reward``. The framework's serializer strips ``reward`` from
    the observation payload, so over the wire ``observation.reward`` is always the
    default 0.0; only ``StepResult.reward`` carries the weighted aggregate.

    The eight reward components are also mirrored in the declared ``components``
    field. The base ``metadata`` dict is stripped by the framework's HTTP
    serialization layer, so ``components`` is what survives the wire; the typed
    client re-populates ``metadata`` from it on the way back.
    """

    prompt: str = Field("", description="Full prompt the policy must answer.")
    answer_to_defend: str = Field(
        "", description="The answer the policy advocates for."
    )
    item_id: str = Field("", description="Source QuALITY article id.")
    reward: float = Field(
        0.0,
        description="In-process weighted aggregate. Stripped over the wire — read "
        "StepResult.reward after a step instead.",
    )
    done: bool = Field(False, description="Whether the episode has ended.")
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Eight reward components (mirror of metadata; survives HTTP).",
    )
    error: str = Field(
        "",
        description="Diagnostic message (e.g. step-before-reset); survives serialization.",
    )
