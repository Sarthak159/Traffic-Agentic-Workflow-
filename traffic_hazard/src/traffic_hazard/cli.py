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

    windows = sub.add_parser(
        "windows", help="extract causal observation windows from a local video"
    )
    windows.add_argument("--video", type=Path, required=True)
    windows.add_argument("--config", type=Path, required=True)
    windows.add_argument("--out", type=Path, required=True)
    windows.add_argument(
        "--source-id", required=True, help="stable logical camera/session ID, e.g. camera-dev-01"
    )
    windows.add_argument("--save-frames", action="store_true", help="write private resized JPEGs")
    windows.add_argument("--contact-sheet", action="store_true", help="write first/last sheets")
    windows.add_argument("--overwrite", action="store_true", help="allow a non-empty --out")
    windows.set_defaults(func=cmd_windows)

    probe = sub.add_parser(
        "probe", help="run the model capability probe against the configured provider"
    )
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--video", type=Path, help="probe on the first window of a local video")
    probe.add_argument("--source-id", help="logical source ID (required with --video)")
    probe.add_argument("--out", type=Path, help="write probe_report.json to this directory")
    probe.add_argument("--overwrite", action="store_true", help="allow a non-empty --out")
    probe.add_argument("--repeats", type=int, default=5, help="latency attempts (1-100)")
    probe.set_defaults(func=cmd_probe)
    return parser


def cmd_probe(args: argparse.Namespace) -> int:
    from .probe import first_video_window, run_probe, synthetic_probe_window
    from .providers.factory import create_provider
    from .storage.artifacts import OutputExistsError, prepare_output_dir, write_json_atomic
    from .video.reader import VideoOpenError

    cfg = _load(args.config)
    if cfg is None:
        return EXIT_USAGE
    apply_tracing_policy(cfg)
    if not 1 <= args.repeats <= 100:
        print("error: --repeats must be between 1 and 100", file=sys.stderr)
        return EXIT_USAGE
    if args.video is not None and not args.source_id:
        print("error: --source-id is required with --video", file=sys.stderr)
        return EXIT_USAGE
    if args.video is not None:
        try:
            window = first_video_window(args.video, cfg, args.source_id)
        except VideoOpenError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        if window is None:
            print("error: the video yields no full observation window", file=sys.stderr)
            return EXIT_USAGE
    else:
        window = synthetic_probe_window(cfg)
    out = None
    if args.out is not None:
        # Created only once a window exists, so a refused run leaves no directory behind.
        try:
            out = prepare_output_dir(args.out, args.overwrite)
        except OutputExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE

    report = asyncio.run(run_probe(cfg, create_provider(cfg.model), window, args.repeats))
    if report["simulated"]:
        print(f"NOTICE            {report['notice']}")
    print(f"provider/model    {report['provider']} / {report['effective_model_ids']}")
    for probe in report["probes"]:
        print(f"{probe['status']:<14}    {probe['name']}: {probe['detail']}")
    latency = report["latency"]
    print(f"latency           p50={latency['p50_s']} s, p95={latency['p95_s']} s")
    if out is not None:
        write_json_atomic(out / "probe_report.json", report)
        print(f"report            {out / 'probe_report.json'}")
    failed = any(p["status"] == "failed" for p in report["probes"])
    return EXIT_FAILED if failed else EXIT_OK


def cmd_windows(args: argparse.Namespace) -> int:
    from .storage.artifacts import OutputExistsError
    from .video.extract import ExtractionError, ExtractionOptions, extract_windows

    cfg = _load(args.config)
    if cfg is None:
        return EXIT_USAGE
    apply_tracing_policy(cfg)
    options = ExtractionOptions(
        save_frames=args.save_frames, contact_sheets=args.contact_sheet, overwrite=args.overwrite
    )
    try:
        summary = extract_windows(
            args.video, cfg, args.out, source_id=args.source_id, options=options
        )
    except (ExtractionError, OutputExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(f"status            {summary['status']}")
    source = summary["source"]
    print(f"source            {source['source_id']} (sha256 {source['sha256'][:16]}…)")
    print(
        f"frames            {summary['frames_accepted']} accepted, "
        f"{summary['first_frame_s']} – {summary['last_frame_s']} s"
    )
    skipped = len(summary["skipped_cutoffs"])
    print(f"windows           {summary['windows_emitted']} emitted, {skipped} skipped")
    print(f"issues            {summary['issues'] or 'none'}")
    for sheet in summary["contact_sheets"]:
        print(f"contact sheet     {args.out / sheet}")
    print(f"output            {args.out / 'windows.jsonl'}")
    return EXIT_OK if summary["status"] == "ok" else EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
