# Build status

Current phase: 2 (fake gate passed; real gate blocked)
Repository commit (base for Phase 0): b2772bf00ae461c66c469ce2b599295445df1612 (branch `traffic-agent-workflow`)
Last verified environment: macOS 27.0.1 arm64, CPython 3.12.11 (uv 0.11.3), openai-agents 0.25.0 (editable, repo root), pydantic 2.14.4, PyAV 16.1.0, numpy 2.5.5, pillow 12.3.0

| Phase | Status (pending/in progress/blocked/passed) | Evidence |
|---|---|---|
| 0 | passed (offline) | see Phase 0 checkpoint below |
| 1 | passed (offline, synthetic footage) | see Phase 1 checkpoint below |
| 2 | fake gate passed; real gate blocked (no Qwen access; user approved fake-only work) | see Phase 2 checkpoint below |
| 3–11 | pending | not run |

## Phase 0 checkpoint (2026-11-22)

- Files changed: new `traffic_hazard/` directory only (pyproject.toml, uv.lock, README.md, EXEC_PLAN.md, BUILD_STATUS.md, .env.example, .gitignore, data/README.md, configs/{defaults,local_fake}.yaml, src/traffic_hazard/{cli,config,contracts,hashing,selftest}.py, src/traffic_hazard/providers/{base,fake}.py, tests/). No SDK source changes; root pyproject.toml/uv.lock untouched.
- Commands actually run (from `traffic_hazard/` unless noted):
  - `uv lock --python 3.12` → Resolved 53 packages.
  - `uv sync --python 3.12 --extra dev` → OK.
  - `uv run traffic-hazard doctor` → exit 0; reports versions, PyAV codecs h264/mpeg4/mjpeg/ffv1 available, env vars "not set", "sdk tracing disabled (no-op)".
  - `uv run traffic-hazard config-check --config configs/local_fake.yaml` → exit 0, fingerprint 665f97ad…
  - `uv run traffic-hazard self-test` → "self-test passed: 10/10", exit 0.
  - `uv run pytest -q tests` → 42 passed (initial); after review fixes the Phase 0 test files give 59 passed (`tests/unit/test_config.py tests/unit/test_contracts.py tests/integration/test_fake_provider.py tests/integration/test_cli_offline.py`).
  - Fresh environment: `UV_PROJECT_ENVIRONMENT=<scratch>/freshenv uv sync --frozen --extra dev` then importing `agents` and `traffic_hazard` → OK (openai-agents 0.25.0); fresh-env self-test 10/10.
  - From repo root: `uv run ruff check traffic_hazard` → All checks passed; `uv run ruff format --check traffic_hazard` → already formatted.
- Gate:
  - [x] Fresh environment can import SDK and application.
  - [x] Config validation fails on unknown or impossible settings (tests/unit/test_config.py, 29 rejection cases).
  - [x] Fake provider returns deterministic typed result (tests/integration/test_fake_provider.py, self-test).
  - [x] No network calls in offline tests (autouse guard on `socket.connect`/`connect_ex`/`create_connection` and non-loopback `getaddrinfo` in tests/conftest.py, verified by test_network_guard_is_active; `gethostbyname` is not patched); SDK tracing disabled by default (verified through public `agents.trace()` returning a no-op trace).
  - [x] BUILD_STATUS.md records actual commands and outcomes.
- Independent review (ordinary tier: new branch-local app, no SDK/durable boundary): one fresh-context reviewer. Cycle 1 found P2×1 and P3×7 (YAML duplicate keys and base-60 numbers accepted; `true` accepted as a float; non-UTF-8 config crashed the CLI; `.env.example` ignored by the root `.env*` pattern; failed Assessments could carry claims; fake specialist output set the app-owned `status`; the network guard missed DNS lookups and `connect_ex`; the self-test default config depended on the working folder). All were fixed. The adaptive budget now requires `max_total_model_calls >= max_specialist_calls + 2`, and `reject_out_of_order` is fixed to `true`. Cycle 2 confirmed every fix and found P3×1 (an unhashable YAML key crashed the CLI). That fix, plus exact-type checks on `Literal` config fields, landed after cycle 2 and is included in the Phase 1 review. 2 of 6 review revisions used.
- Real model tested? no.
- Decisions / deviations:
  - Stayed on existing branch `traffic-agent-workflow` (branch creation needs explicit approval per repo CLAUDE.md).
  - Used uv (the fork's tool) with a separate app lock instead of venv/pip.
  - `configs/defaults.yaml` is the only source of default values; config models have no defaults, so missing keys fail.
  - Only `provider: fake` is accepted until Phase 2; `qwen_hosted.yaml`/`qwen_local.yaml` and `prompts/` are deferred to Phases 2–3.
  - Fake provider covers assessment and specialist outputs plus malformed/timeout/error; the route action schema is deferred to Phase 5 where it is defined.
  - A CI workflow job was not added: changing `.github/workflows` is repository CI configuration and was not requested; the offline command is `uv run pytest -q tests` from `traffic_hazard/`.
- Blockers: none for Phase 0.
- Next smallest task: Phase 1 video reader and causal window builder.
- Rollback config or commit: revert the Phase 0 commit (additive directory only).

## Phase 1 checkpoint (2026-11-22)

- Files changed: `src/traffic_hazard/cli.py` (new `windows` command); new `src/traffic_hazard/video/{reader,windows,encoding,extract,synthetic}.py`, `runtime/clock.py`, `storage/artifacts.py`, `scripts/make_fixture.py`, `data/README.md`, tests `tests/unit/{test_windows,test_clock}.py`, `tests/contract/test_causality.py`, `tests/integration/{test_video_reader,test_windows_cli}.py`, `tests/fixtures/video_helpers.py`.
- Commands actually run (from `traffic_hazard/`):
  - `uv run pytest -q tests` → 106 passed (after review fixes).
  - `uv run python scripts/make_fixture.py --out data/dev/fixture.mp4` → 12 s synthetic clip, 30 FPS, libx264, 640×360.
  - `uv run traffic-hazard windows --video data/dev/fixture.mp4 --config configs/local_fake.yaml --out runs/dev_windows --source-id camera-dev-01 --contact-sheet --save-frames` → status ok, 360 frames (0.0–11.966667 s), 7 windows (cutoffs 5–11 s), 0 skipped, no decode issues.
  - From repo root: `uv run ruff check traffic_hazard` and `ruff format --check traffic_hazard` → clean.
- Gate:
  - [x] At every cutoff all selected frame timestamps lie in `(t-5, t]` (contract validator on `ObservationWindow`; `test_all_selected_frames_lie_in_the_causal_interval`; streaming result equals a brute-force oracle over 8 random variable-rate streams).
  - [x] Changing frames after the cutoff leaves the window hash unchanged (in-memory test and real ffv1 files differing only after 8 s; prefix test: truncating the stream after t reproduces window t exactly).
  - [x] Repeat extraction gives identical IDs and hashes (libx264 file decoded twice; CLI run twice gives byte-identical `windows.jsonl`).
  - [x] Variable-rate, short, missing/corrupt, duplicate-timestamp and end-of-stream fixtures pass (`test_video_reader.py`, `test_windows.py`, `test_windows_cli.py`; truncated and byte-corrupted files decode without crashing).
  - [x] Manual inspection: both contact sheets of `runs/dev_windows` checked by eye. 20 frames each, labels strictly increasing, burnt-in times equal the frame IDs, all inside (t-5, t]; the last source frame (11.967 s) is correctly excluded from cutoff 11.
- Independent review (ordinary tier): fresh-context reviewer, 2 cycles. Cycle 1 found P2×1 (the default content-hash source ID made window IDs depend on footage after the cutoff), P3×1 (sub-microsecond config values could crash window validation) and P3×1 (`--out` naming a file raised a traceback). All were fixed with tests. Cycle 2 confirmed the fixes and found P3×1 (a stale command in EXEC_PLAN.md, now fixed). The cycle 2 review also covered the post-review Phase 0 config changes (the unhashable-key check and exact-type checks on `Literal` fields): no findings.
- Real model tested? no (not in Phase 1 scope).
- Decisions / deviations:
  - `transport: native_video` is rejected with a clear error; native video clips are deferred to Phase 2, when the provider's video input format is known.
  - Cutoffs lie on the grid `n × interval` of source time, and the first needs a full window after the first frame.
  - Times are exact fractions, quantized to integer microseconds; frame IDs are `f-<seconds>.<6 digits>`.
  - Frame hashes cover the decoded and resized RGB pixels (`rgb24-bilinear-v1`), not JPEG bytes.
  - Quality flags: `gap_ticks:N` (no frame inside the window for a tick), `repeated_frame_ticks:N`, `stale_ticks:N` (the selected frame is at least one sampling interval older than its tick), `decode_issue:<kind>`.
  - Untimed decode issues are attributed to the next accepted frame, so they cannot leak into an earlier window.
  - Lossy inter-frame codecs (h264) can change the pixels of earlier B-frames when later footage is re-encoded. That is an encoding effect, not pipeline leakage, so the real-file future-leakage test uses intra-only ffv1.
  - `--overwrite` deletes only this command's own artifacts.
  - `--source-id` is required (a deviation from the spec's example command). A whole-file hash default would make window IDs depend on footage after the cutoff, and a file-name default could leak labels. The file SHA-256 is recorded only in `extraction_summary.json`.
  - `window_seconds` and `assessment_interval_seconds` must be whole microseconds, so exact selection and the written times cannot disagree.
  - Known limitations: a large backward timestamp jump (concatenated segments, PTS reset) drops all later footage as `out_of_order`. With B-frame codecs, an undecodable packet's `corrupt` flag may be attributed to a frame at or before t, as decode order differs from presentation order (an inherent codec limit; real-file leakage tests use intra-only ffv1). Missing and duplicate PTS are covered by in-memory tests only. Frame hashes cover resized pixels, not JPEG bytes; the JPEG quality is recorded in the summary.
- Blockers: none for the offline gate. Real footage is not yet authorized; only synthetic clips were used.
- Next smallest task: Phase 2, the provider abstraction and capability probe (fake path; the real probe is blocked on Qwen access).
- Rollback config or commit: revert the Phase 1 commit; Phase 0 stays usable.

## Phase 2 checkpoint, fake path only (2026-11-22)

- Scope approved by the user: fake provider only, with no real adapter code and no live calls.
- Files changed:
  - New: `src/traffic_hazard/providers/{output,guard,factory}.py`, `src/traffic_hazard/probe.py`, `tests/contract/test_provider.py`, `tests/integration/test_probe.py`.
  - Changed: `providers/fake.py` (fake output now goes through the shared strict parser; new `raw`/`raise` kinds and contradictory/future-frame/malformed fixtures), `contracts.py` (`check_judgment` shared by `Assessment` and `AssessmentOutput`), `cli.py` (`probe` command), `selftest.py`, `README.md`.
- Commands actually run (from `traffic_hazard/`):
  - `uv run pytest -q tests` → 137 passed (after review fixes).
  - `uv run traffic-hazard probe --config configs/local_fake.yaml --out runs/probe_fake` → exit 0. connectivity, json_assessment, json_specialist and cost_latency passed. still_image, ordered_frames, reversed_frames, control_frames and tool_selection are not_evaluable; native_video was skipped. 7 attempts in total.
  - `uv run traffic-hazard self-test` → 10/10.
  - From repo root: `uv run ruff check traffic_hazard` and `ruff format --check traffic_hazard` → clean.
- Gate:
  - [x] Fake provider tests pass for success, malformed output, timeout and explicit failures (tests/contract/test_provider.py, tests/integration/test_fake_provider.py). Contradictory and out-of-window outputs are rejected by output validation, and the guard turns raising, hanging and wrong-type providers into typed errors.
  - [ ] Real model consumes time-ordered visual observations and returns valid typed information on authorized data. **Blocked:** no Qwen model ID, endpoint or credentials, and no authorized footage.
  - [ ] Record effective model ID/revision, transport, tool support, output mode, cost, time and failures for a real model. The probe report already records these fields; they are not yet measured on a real model.
  - [x] No claims of accurate hazard anticipation. Fake probe reports are marked `simulated: true` with a notice, and media and tool probes are never reported as passed.
- Independent review (ordinary tier): fresh-context reviewer. Cycle 1 found P2×2 and P3×1: some malformed responses were classified as `provider_error`; the synthetic probe crashed for intervals of 2–4 s; and model-controlled key names leaked into errors and the probe report. All were fixed with tests, together with three uncertainty items: late results, guard-measured latency, and `1e999`. Cycle 2 confirmed every fix with no new findings; it also fuzzed 504 valid video configs through `synthetic_probe_window`, and all produced a window. 2 of 6 review revisions used.
- Real model tested? no.
- Decisions / deviations:
  - The model never supplies app-owned fields. `AssessmentOutput` excludes window ID, status, horizon and record IDs; `SpecialistOutput` uses `sufficient: bool`, which the app maps to `insufficient_evidence`. Mapping into `Assessment`/`SpecialistEvidence` records happens in Phase 3.
  - Output parsing is strict: exactly one JSON object with outer whitespace only. Markdown fences, prose, duplicate keys, NaN/Infinity/overflowing floats, oversized, too-deep or huge-integer responses all end up as `malformed_output` end to end through the guard, with no repair. If a real Qwen endpoint wraps JSON in fences, any repair must be added explicitly and counted.
  - `guarded_generate` enforces `timeout_s` (+0.25 s grace) and treats a result that arrives late (after cancellation was swallowed) as a timeout. Its own monotonic measurement replaces the provider's `elapsed_s`, and error messages keep only the exception type name. Model-controlled key names never appear in validation errors (`<extra>`), and at most 8 errors are reported.
  - SDK `Runner`/tool integration is deferred to Phase 3/5, once real tool-call support is known. `allow_retries` is not acted on: no retries happen, and every attempt is counted.
  - No `--allow-live` flag yet: no live provider exists, and the config accepts only `fake`.
- Carry-forward items for the real adapter and Phase 3 (not reachable with the fake provider):
  - Move "drop `raw_text` over 100k chars" from `fake.py` into a shared helper, so a real adapter cannot hit a `provider_error` on oversized output.
  - Have the guard re-raise caller cancellation that a provider swallowed (`asyncio.current_task().cancelling()`).
  - Phase 3 must make sure that persisting model text containing lone UTF-16 surrogates (`"\ud800"`) cannot fail record writes.
- Blockers: the real Phase 2 gate needs the exact Qwen model ID, endpoint, credentials, a decision on footage handling, and approval for paid calls.
- Next smallest task: Phase 3, the single-agent baseline on the fake provider (prompts/main_v1.md, normalizing output into immutable `AssessmentRecord`s, and a `replay --mode single` command).
- Rollback config or commit: revert the Phase 2 commit; Phases 0–1 are unaffected apart from the self-test and fake-provider tests updated in it.
