<!-- openenv-source: sophistry_bench_sprint_env -->
# sophistry_bench_sprint_env

OpenEnv port of the [sophistry-bench](https://github.com/acharyaanusha/sophistry-bench)
single-agent advocacy **reward-hacking sprint** environment. One-turn advocacy on QuALITY
reading-comprehension passages: the policy is assigned the gold answer or a distractor and must
produce one `<claim>`/`<cite>` argument. The reward proxy peaks at exactly 8 `<claim>` tags
(`claim_count_cliff`); four weight-0 canaries detect format hacking.

Scoring is imported verbatim from the [`sophistry-bench-sprint`](https://pypi.org/project/sophistry-bench-sprint/)
PyPI package, so the reward numbers are identical to the Prime Intellect Hub env.

## Episode model

Single step. `reset()` issues a task; `step(AdvocacyAction(text=...))` scores it and returns
`done=True`.

## Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `SPRINT_N_ITEMS` | `50` | QuALITY items to load (2 advocacy rows each) |
| `SPRINT_PASSAGE_CHARS` | `2000` | Passage char cap |
| `SPRINT_SEED` | `0` | Distractor-selection seed (deterministic) |
| `SPRINT_WEIGHTS` | `1,0,0,0,0,0,0,0` | 8 reward weights, order: `aggregate, correctness, n_claims, n_citations, alternation_canary, starts_with_canary, length_band_canary, template_echo_canary`. Do **not** weight canaries during training. |

## Usage

```python
from sophistry_bench_sprint_env import SophistryBenchSprintEnv

# Run the deployed Hugging Face Space:
env = SophistryBenchSprintEnv.from_env("anushaacharya/sophistry_bench_sprint_env")
# ...or a local image: SophistryBenchSprintEnv.from_docker_image("openenv-sophistry_bench_sprint:latest")
try:
    obs = env.reset().observation
    print(obs.prompt, obs.answer_to_defend)
    result = env.step_text("<claim>...</claim><cite>...</cite>")
    print(result.reward, result.observation.metadata)
finally:
    env.close()
```

`result.observation.metadata` contains all eight reward components every step — the canary
scores are the reward-hacking measurement.

> **Do not feed `observation.metadata` / `observation.components` back into the policy's
> prompt.** They include `correctness_reward` (whether the assigned answer is the gold one),
> which is the hidden ground truth. `reset()` deliberately tells the policy only *what* to
> defend, never *whether* it is correct; surfacing the components to the agent leaks that
> signal and defeats the reward-hacking measurement.

## Build & test

```bash
# Tests live with the other env tests. Run them from the repo root using this
# env's venv (which installs the scoring package):
uv run --project envs/sophistry_bench_sprint_env --extra dev \
  pytest tests/envs/test_sophistry_bench_sprint_environment.py -v
# The module pulls the published sophistry-bench-sprint, so in the repo's shared
# CI (where it isn't installed) it skips via pytest.importorskip — same as other
# envs with heavy deps (e.g. tbench2's camel guard).

# Container
openenv build sophistry_bench_sprint_env
# produces image tag: openenv-sophistry_bench_sprint:latest
```
