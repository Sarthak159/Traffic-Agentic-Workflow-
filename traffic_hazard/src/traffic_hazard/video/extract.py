"""Offline window extraction from a local video file into private run artifacts.

Outputs in the run directory:

- `windows.jsonl`: one immutable `ObservationWindow` per emitted cutoff, in cutoff order.
- `extraction_summary.json`: source hash, encoding version, config fingerprints, counts, issues
  and skipped cutoffs.
- `media/<sha256>.jpg`: resized frames, only when frame saving is enabled.
- `contact_sheets/*.png`: labeled grids of the first and last windows, only when requested.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..contracts import SOURCE_ID_PATTERN
from ..hashing import file_sha256, fingerprint
from ..storage.artifacts import (
    JsonlWriter,
    prepare_output_dir,
    write_bytes_atomic,
    write_json_atomic,
)
from .encoding import ENCODING_VERSION, JPEG_QUALITY, contact_sheet, jpeg_bytes
from .reader import US_PER_S, PyAVVideoSource, VideoOpenError
from .windows import WindowBuilder, WindowResult

SUMMARY_SCHEMA_VERSION = "1"
WINDOWS_FILE = "windows.jsonl"
SUMMARY_FILE = "extraction_summary.json"


class ExtractionError(RuntimeError):
    """The extraction could not start or the input is unusable."""


@dataclass(frozen=True)
class ExtractionOptions:
    save_frames: bool = False
    contact_sheets: bool = False
    overwrite: bool = False


def extract_windows(
    video: Path,
    cfg: AppConfig,
    out: Path,
    *,
    source_id: str,
    options: ExtractionOptions | None = None,
    on_window: Callable[[WindowResult], None] | None = None,
) -> dict[str, Any]:
    """Decode `video`, write causal windows and a summary to `out`, and return the summary.

    `source_id` is a stable logical camera/session ID chosen by the caller. It must not be derived
    from the file's content (a whole-file hash changes when later footage changes, which would leak
    future data into every window ID) or from a possibly label-revealing file name.
    """
    if cfg.video.transport != "frames":
        raise ExtractionError(
            "transport 'native_video' is not implemented yet (planned with Phase 2)"
        )
    if not video.is_file():
        raise ExtractionError(f"video file not found: {video}")
    source_sha = file_sha256(video)
    if not re.fullmatch(SOURCE_ID_PATTERN, source_id):
        raise ExtractionError(f"invalid source ID {source_id!r}")
    options = options or ExtractionOptions()
    out = prepare_output_dir(out, options.overwrite)
    _remove_previous_artifacts(out)
    save_frames = options.save_frames or cfg.logging.save_private_frames

    builder = WindowBuilder(cfg.video, source_id)
    first: WindowResult | None = None
    last: WindowResult | None = None
    written_media: set[str] = set()
    status = "ok"
    error: str | None = None

    with JsonlWriter(out / WINDOWS_FILE) as writer:

        def handle(result: WindowResult) -> None:
            nonlocal first, last
            writer.write(result.window.model_dump(mode="json"))
            if save_frames:
                for sha, image in result.images.items():
                    if sha not in written_media:
                        write_bytes_atomic(out / "media" / f"{sha}.jpg", jpeg_bytes(image))
                        written_media.add(sha)
            first = first or result
            last = result
            if on_window is not None:
                on_window(result)

        try:
            for item in PyAVVideoSource(video):
                for result in builder.feed(item):
                    handle(result)
        except VideoOpenError as exc:
            status, error = "failed", str(exc)
        for result in builder.finish():
            handle(result)

    if status == "ok" and writer.count == 0:
        if builder.frames_seen == 0:
            status = "no_frames"
        elif not builder.skipped:
            status = "insufficient_history"
        else:
            status = "no_usable_windows"

    sheets = []
    if options.contact_sheets:
        for result in {id(r): r for r in (first, last) if r is not None}.values():
            sheets.append(_write_contact_sheet(out, result))

    video_cfg = cfg.video.model_dump(mode="json")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "error": error,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "app_version": _version("traffic-hazard"),
        "decoder": f"PyAV {_version('av')}",
        "source": {"source_id": source_id, "sha256": source_sha, "file_name": video.name},
        "jpeg_quality": JPEG_QUALITY,
        "encoding_version": ENCODING_VERSION,
        "video_config": video_cfg,
        "video_config_fingerprint": fingerprint(video_cfg),
        "config_fingerprint": cfg.fingerprint(),
        "frames_accepted": builder.frames_seen,
        "first_frame_s": _seconds(builder.first_frame_us),
        "last_frame_s": _seconds(builder.last_frame_us),
        "issues": dict(sorted(builder.issues.items())),
        "windows_emitted": writer.count,
        "skipped_cutoffs": [{"cutoff_s": s.cutoff_s, "reason": s.reason} for s in builder.skipped],
        "media_saved": len(written_media),
        "contact_sheets": sheets,
    }
    write_json_atomic(out / SUMMARY_FILE, summary)
    return summary


def _remove_previous_artifacts(out: Path) -> None:
    """On overwrite, delete only artifacts this command owns so no stale output is mixed in."""
    for name in (WINDOWS_FILE, SUMMARY_FILE):
        (out / name).unlink(missing_ok=True)
    for name in ("media", "contact_sheets"):
        if (out / name).is_dir():
            shutil.rmtree(out / name)


def _write_contact_sheet(out: Path, result: WindowResult) -> str:
    window = result.window
    images = [result.images[f.content_sha256] for f in window.frames]
    labels = [f"{f.frame_id}" for f in window.frames]
    sheet = contact_sheet(images, labels)
    name = f"contact_sheets/cutoff_{window.cutoff_s:010.3f}_{window.window_id}.png"
    import io

    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    write_bytes_atomic(out / name, buffer.getvalue())
    return name


def _seconds(time_us: int | None) -> float | None:
    return None if time_us is None else time_us / US_PER_S


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"
