# Build status

Current phase: 1 (in progress)
Repository commit (base for Phase 0): b2772bf00ae461c66c469ce2b599295445df1612 (branch `traffic-agent-workflow`)
Last verified environment: macOS 27.0.1 arm64, CPython 3.12.11 (uv 0.11.3), openai-agents 0.25.0 (editable, repo root), pydantic 2.14.4, PyAV 16.1.0, numpy 2.5.5, pillow 12.3.0

| Phase | Status (pending/in progress/blocked/passed) | Evidence |
|---|---|---|
| 0 | passed (offline) | see Phase 0 checkpoint below |
| 1 | pending | not run |
| 2 | pending | fake and real gates tracked separately; real gate blocked: no Qwen access yet |
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
