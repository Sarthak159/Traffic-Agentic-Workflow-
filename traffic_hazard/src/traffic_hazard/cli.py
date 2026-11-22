"""`traffic-hazard` command-line entry point. Every command here runs offline."""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from .config import AppConfig, ConfigError, load_config

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
# `traffic_hazard/` (the app root) for the editable install; default paths resolve from here.
APP_ROOT = Path(__file__).resolve().parents[2]
SECRET_ENV_VARS = ("QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL_ID", "OPENAI_API_KEY")


def apply_tracing_policy(cfg: AppConfig | None) -> bool:
    """Disable SDK remote tracing unless the config enables it; return the disabled state."""
    from agents import set_tracing_disabled

    disabled = cfg is None or not cfg.logging.sdk_remote_tracing
    set_tracing_disabled(disabled)
    return disabled


def sdk_tracing_is_noop() -> bool:
    """Observe SDK tracing through public API: a disabled provider yields a no-op trace."""
    from agents import trace

    return trace("traffic-hazard-probe").trace_id == "no-op"


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def cmd_doctor(_: argparse.Namespace) -> int:
    apply_tracing_policy(None)
    print(f"python            {platform.python_version()} ({sys.executable})")
    print(f"platform          {platform.platform()}")
    for dist in ("traffic-hazard", "openai-agents", "pydantic", "pyyaml", "av", "numpy", "pillow"):
        print(f"{dist:<17} {_version(dist)}")
    try:
        import av

        codecs = {name: name in av.codecs_available for name in ("h264", "mpeg4", "mjpeg", "ffv1")}
        print(f"decoder           PyAV {av.__version__}; codecs {codecs}")
    except Exception as exc:
        print(f"decoder           UNAVAILABLE ({type(exc).__name__})")
    print("device            cpu (GPU probing is not part of the offline scaffold)")
    for name in SECRET_ENV_VARS:
        print(f"env {name:<15}{'set' if os.environ.get(name) else 'not set'}")
    print(f"sdk tracing       {'disabled (no-op)' if sdk_tracing_is_noop() else 'ENABLED'}")
    return EXIT_OK


def _load(path: Path) -> AppConfig | None:
    try:
        return load_config(path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return None


def cmd_config_check(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_USAGE
    wf, video = cfg.workflow, cfg.video
    print(f"config OK: {args.config}")
    print(f"fingerprint       {cfg.fingerprint()}")
    print(f"provider/model    {cfg.model.provider} / {cfg.model.model_id}")
    print(f"workflow          {cfg.workflow.mode}; pool={list(cfg.workflow.available_specialists)}")
    print(
        f"budget            calls<={wf.max_total_model_calls}, "
        f"specialist calls<={wf.max_specialist_calls}, unique<={wf.max_unique_specialists}"
    )
    print(
        f"windows           {video.window_seconds}s @ {video.sample_fps} FPS "
        f"(<= {video.max_frames} frames)"
    )
    return EXIT_OK


def cmd_self_test(args: argparse.Namespace) -> int:
    from .selftest import run_self_test

    cfg = _load(args.config)
    if cfg is None:
        return EXIT_USAGE
    apply_tracing_policy(cfg)
    return EXIT_OK if asyncio.run(run_self_test(cfg)) else EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic-hazard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "doctor", help="report environment, decoder and tracing state (no secrets)"
    ).set_defaults(func=cmd_doctor)
    check = sub.add_parser("config-check", help="validate a configuration file")
    check.add_argument("--config", type=Path, required=True)
    check.set_defaults(func=cmd_config_check)

    selftest = sub.add_parser(
        "self-test", help="offline checks with the deterministic fake provider"
    )
    selftest.add_argument("--config", type=Path, default=APP_ROOT / "configs" / "local_fake.yaml")
    selftest.set_defaults(func=cmd_self_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
