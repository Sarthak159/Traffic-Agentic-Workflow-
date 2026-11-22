# Traffic hazard app: Phase 0 scaffold and Phase 1 causal video windows

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must stay up to date as work proceeds. It is maintained in accordance with `PLANS.md` at the repository root. The research specification it implements is `TRAFFIC_AGENT_PIPELINE.md` at the repository root (sections 3, 4, 5 and 16).

## Purpose / Big Picture

After this work a developer can, from `traffic_hazard/`, install an isolated application environment, run `traffic-hazard doctor`, validate a YAML configuration with `traffic-hazard config-check`, run an offline `traffic-hazard self-test` that exercises a deterministic fake model provider, and turn a local video file into causal, timestamped, immutable observation windows with `traffic-hazard windows`. A window is the set of up to 20 frames sampled at 4 FPS from the five seconds of source time ending at a cutoff `t`; "causal" means no frame later than `t` can ever influence a window. Nothing contacts a network service and no model is called.

## Progress

- [x] (2026-11-22) Read spec, repo instructions, PLANS.md, implementation-strategy and implementation-final-review skills; confirmed scope with user (Phases 0 and 1 only, commit per phase, fake provider only, synthetic footage only).
- [x] (2026-11-22) Phase 0: scaffold package, configuration loader/validator, domain contracts, fake provider, CLI (`doctor`, `config-check`, `self-test`), offline tests with network guard, BUILD_STATUS.md.
- [x] (2026-11-22) Phase 0: independent review (2 cycles, all findings fixed), gate recorded, commit.
- [ ] Phase 1: video reader (PyAV), encoding, window builder, replay clock, `windows` command, synthetic fixtures, causality tests.
- [ ] Phase 1: independent review, manual contact-sheet inspection, gate recorded, commit.

## Surprises & Discoveries

- Observation: PyYAML (YAML 1.1) parses `cache_mode: off` as False, `1:30` as 90 and `010` as octal, and silently keeps the last of duplicate keys.
  Evidence: reviewer probes; fixed by `_StrictLoader` in config.py with regression tests.
- Observation: the repository root `.gitignore` has `.env*`, which hid `traffic_hazard/.env.example`.
  Evidence: `git ls-files --others --exclude-standard`; fixed with `!.env.example` in traffic_hazard/.gitignore.
- Observation: Matroska stores millisecond timestamps, so a 30 FPS ffv1/mkv fixture decodes 1/30 s as 33 ms; mp4 with a 1/30000 time base is exact. The mpeg4 encoder rejects time-base denominators above 65535.
  Evidence: tests/integration/test_video_reader.py.

## Decision Log

- Decision: Untimed decode issues (for example undecodable packets) are attributed to the next accepted frame, not the previous one.
  Rationale: Attributing them to the previous frame let a problem after cutoff t change the flags and hash of the window for t, which is future leakage.
  Date/Author: 2026-11-22 / Claude.
- Decision: `video.reject_out_of_order` only accepts `true`.
  Rationale: The streaming builder cannot accept a frame earlier than one already seen without breaking causality; reordering would need lookahead.
  Date/Author: 2026-11-22 / Claude.

- Decision: Work on the existing branch `traffic-agent-workflow` instead of creating `traffic/adaptive-pipeline`.
  Rationale: Repository CLAUDE.md requires explicit approval to create branches; the current branch is already dedicated to this work.
  Date/Author: 2026-11-22 / Claude.
- Decision: `traffic_hazard/` is a standalone uv project with its own `pyproject.toml` and `uv.lock`, depending on the SDK at `..` as an editable path source. Python 3.12.
  Rationale: Spec asks for application-specific dependencies and an isolated environment; this keeps the root SDK `pyproject.toml`/`uv.lock` untouched and avoids conflicting lockfiles. 3.12 is the newest version the spec names and is installed locally.
  Date/Author: 2026-11-22 / Claude.
- Decision: Qwen config files (`qwen_hosted.yaml`, `qwen_local.yaml`) and real providers are deferred to Phase 2.
  Rationale: Out of approved scope; a config naming an unimplemented provider would fail `config-check`.
  Date/Author: 2026-11-22 / Claude.
- Decision: Frame and window times are computed with exact rational arithmetic (`fractions.Fraction` from PTS × time base) and only rounded to microseconds for output.
  Rationale: Avoids floating-point drift in sampling ticks and makes IDs/hashes deterministic.
  Date/Author: 2026-11-22 / Claude.

## Outcomes & Retrospective

(to be written at each phase gate)

## Context and Orientation

The repository is a fork of the OpenAI Agents SDK (`src/agents/`, version 0.25.0), managed with `uv`. The SDK is not modified. The new application lives entirely in `traffic_hazard/`: package source under `traffic_hazard/src/traffic_hazard/`, tests under `traffic_hazard/tests/`, YAML configs under `traffic_hazard/configs/`. Private data (`traffic_hazard/data/`) and outputs (`traffic_hazard/runs/`) are gitignored. Root `ruff check` lints the whole repository, so app code must satisfy the root ruff configuration; root pytest collects only `tests/`, so app tests run via the app environment.

Terms: a "cutoff" is the source-time instant `t` of an assessment; a "sampling tick" is one of the instants `t, t-0.25, ..., t-4.75` at which the latest frame at or before the tick is selected; "source time" is seconds since the first decoded frame's presentation timestamp (PTS), not wall-clock time.

## Plan of Work

Phase 0 creates `pyproject.toml`, `README.md`, `.env.example`, `.gitignore`, `BUILD_STATUS.md`, `configs/defaults.yaml` and `configs/local_fake.yaml`, and the modules `config.py` (strict Pydantic models, deep-merge of an overlay onto defaults, cross-field validation), `contracts.py` (domain schemas from spec section 3 with validators), `providers/base.py` (typed `ModelRequest`, `ModelResult`, `ModelError`, `ModelProvider` protocol), `providers/fake.py` (deterministic scripted provider with call counter, delay, malformed output, timeout and error modes), and `cli.py`. Tests include a session-wide socket guard so offline tests fail loudly on any network attempt.

Phase 1 adds `video/reader.py` (PyAV-backed `VideoSource` plus an in-memory source for tests, normalizing PTS to source seconds and recording corrupt/missing/duplicate/out-of-order frames), `video/encoding.py` (aspect-preserving resize, canonical pixel hash, JPEG encode, contact sheet), `video/windows.py` (streaming `WindowBuilder` emitting `ObservationWindow`s), `runtime/clock.py` (unpaced and paced replay clocks over a monotonic clock), `storage/artifacts.py` (atomic JSON/JSONL writes), and the `windows` CLI command. Synthetic fixture videos are generated by test fixtures with PyAV and by `scripts/make_fixture.py`.

## Concrete Steps

From `traffic_hazard/`:

    uv sync --python 3.12
    uv run traffic-hazard doctor
    uv run traffic-hazard config-check --config configs/local_fake.yaml
    uv run traffic-hazard self-test
    uv run pytest -q tests -m 'not live'
    uv run python scripts/make_fixture.py --out data/dev/fixture.mp4
    uv run traffic-hazard windows --video data/dev/fixture.mp4 --config configs/local_fake.yaml --out runs/dev_windows --contact-sheet

## Validation and Acceptance

Phase 0 gate (spec section 4): fresh environment imports SDK and app; config validation fails on unknown or impossible settings; fake provider returns a deterministic typed result; no network calls in offline tests and SDK tracing disabled by default; BUILD_STATUS.md records actual commands and outcomes.

Phase 1 gate (spec section 5): every selected frame timestamp is within `(t-5, t]`; changing frames after the cutoff leaves the window hash unchanged; repeat extraction gives identical IDs and hashes; variable-rate, short, corrupt/missing, duplicate-timestamp and end-of-stream fixtures pass; one contact sheet inspected manually.

## Idempotence and Recovery

All work is additive under `traffic_hazard/`. Rolling back is `git revert` of the phase commit or deleting the directory. `uv sync` is idempotent. The `windows` command refuses to overwrite a non-empty output directory unless `--overwrite` is given.

## Artifacts and Notes

Command transcripts are recorded in `traffic_hazard/BUILD_STATUS.md`.

## Interfaces and Dependencies

Dependencies: `openai-agents` (path `..`), `pydantic>=2`, `pyyaml`, `av` (PyAV, decoder with exact timestamps), `numpy`, `pillow`; dev: `pytest`, `pytest-asyncio`. Interfaces at the end of Phase 1:

    traffic_hazard.config.load_config(path: Path | None) -> AppConfig
    traffic_hazard.providers.base.ModelProvider.generate(request: ModelRequest) -> Awaitable[ModelResult]
    traffic_hazard.providers.fake.FakeProvider(script: Mapping[str, FakeResponse] | None = None)
    traffic_hazard.video.reader.VideoSource  (iterable of DecodedFrame | DecodeIssue)
    traffic_hazard.video.windows.WindowBuilder(cfg: VideoConfig, source_id: str).feed(frame) / .finish()
    traffic_hazard.contracts.ObservationWindow, FrameRef, Observation, SpecialistEvidence, Assessment
