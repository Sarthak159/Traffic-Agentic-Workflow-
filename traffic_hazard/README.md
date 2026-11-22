# traffic-hazard

Research prototype for adaptive multimodal traffic hazard anticipation on recorded intersection video. It is built inside this OpenAI Agents SDK fork without modifying `src/agents/`. The specification and phase plan are in [`../TRAFFIC_AGENT_PIPELINE.md`](../TRAFFIC_AGENT_PIPELINE.md); current progress is in [`BUILD_STATUS.md`](BUILD_STATUS.md).

**RESEARCH SIMULATION — NOT EMERGENCY DISPATCH.** Alerts are local and simulated only. Default runs are offline, use a deterministic fake model, and keep SDK remote tracing disabled.

## Setup

`traffic_hazard/` is a standalone uv project (its own `pyproject.toml` and `uv.lock`) that installs the SDK from the repository root as an editable path dependency. Run every command from `traffic_hazard/`.

    uv sync --python 3.12 --extra dev
    uv run traffic-hazard doctor
    uv run traffic-hazard config-check --config configs/local_fake.yaml
    uv run traffic-hazard self-test
    uv run pytest -q tests          # `-m 'not live'` is the default; offline tests block network sockets

## Video windows and the model probe

    uv run python scripts/make_fixture.py --out data/dev/fixture.mp4      # synthetic clip, no real footage
    uv run traffic-hazard windows --video data/dev/fixture.mp4 --config configs/local_fake.yaml \
        --out runs/dev_windows --source-id camera-dev-01 --contact-sheet
    uv run traffic-hazard probe --config configs/local_fake.yaml --out runs/probe_fake

`--source-id` is a stable logical camera/session ID. It is required because window IDs must not depend on file contents after the cutoff or on file names. With the fake provider, `probe` checks the harness only: media and tool probes are reported as `not_evaluable`, never as passed.

## Configuration

`configs/defaults.yaml` holds every default value. Other configs inherit it with `extends: defaults.yaml` and override only what they change. Unknown keys, impossible combinations (fixed roles outside the pool, call budgets below the required synthesis call, sampling that exceeds the frame cap), non-finite numbers and YAML 1.1 boolean spellings such as `yes`/`off` are rejected. Secrets come only from the environment (see `.env.example`); `doctor` reports whether they are set, never their values.

## Data handling

`data/` and `runs/` are gitignored and private. Never commit videos, credentials, model weights or `.env`. Do not send footage to a hosted API without explicit authorization.
