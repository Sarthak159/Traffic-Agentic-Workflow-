from __future__ import annotations

import pytest
from pydantic import ValidationError
from traffic_hazard.contracts import (
    Assessment,
    FrameRef,
    Observation,
    ObservationWindow,
    SpecialistEvidence,
    unknown_frame_refs,
)
from traffic_hazard.selftest import synthetic_window

WID = "w-" + "a" * 24


def _frame(t: float, i: int = 1) -> FrameRef:
    return FrameRef(
        frame_id=f"f-{t:.6f}",
        source_time_s=t,
        content_sha256=f"{i:064x}",
        local_ref=f"media/{i}.jpg",
    )


def test_window_interval_rules() -> None:
    ObservationWindow(
        source_id="cam", window_id=WID, start_s=0, cutoff_s=5, frames=(_frame(1), _frame(5, 2))
    )
    with pytest.raises(ValidationError, match="interval"):
        ObservationWindow(
            source_id="cam", window_id=WID, start_s=0, cutoff_s=5, frames=(_frame(5.25),)
        )
    with pytest.raises(ValidationError, match="interval"):
        ObservationWindow(
            source_id="cam", window_id=WID, start_s=1, cutoff_s=5, frames=(_frame(1),)
        )
    with pytest.raises(ValidationError, match="strictly increasing"):
        ObservationWindow(
            source_id="cam", window_id=WID, start_s=0, cutoff_s=5, frames=(_frame(3), _frame(2, 2))
        )
    with pytest.raises(ValidationError, match="unique"):
        ObservationWindow(
            source_id="cam", window_id=WID, start_s=0, cutoff_s=5, frames=(_frame(3), _frame(3))
        )
    with pytest.raises(ValidationError, match="start_s"):
        ObservationWindow(
            source_id="cam", window_id=WID, start_s=6, cutoff_s=5, frames=(_frame(5),)
        )
    with pytest.raises(ValidationError):
        ObservationWindow(source_id="cam", window_id=WID, start_s=0, cutoff_s=5, frames=())
    with pytest.raises(ValidationError, match="extra"):
        ObservationWindow(
            source_id="cam",
            window_id=WID,
            start_s=0,
            cutoff_s=5,
            frames=(_frame(5),),
            label="crash",
        )


@pytest.mark.parametrize(
    "ref", ["/abs/path.jpg", "../escape.jpg", "media/../../x", "https://host/x.jpg", ""]
)
def test_frame_ref_rejects_paths_and_urls(ref: str) -> None:
    with pytest.raises(ValidationError):
        FrameRef(frame_id="f-1.000000", source_time_s=1, content_sha256="0" * 64, local_ref=ref)


def test_frame_ref_rejects_non_finite_time() -> None:
    with pytest.raises(ValidationError):
        FrameRef(
            frame_id="f-1.000000",
            source_time_s=float("nan"),
            content_sha256="0" * 64,
            local_ref="m/x.jpg",
        )


def test_assessment_consistency() -> None:
    obs = Observation(
        text="Vehicle approaches crossing.", frame_ids=("f-5.000000",), kind="observed"
    )
    Assessment(
        window_id=WID,
        status="ok",
        risk_level="high",
        event_state="potential_conflict",
        event_type="vehicle_pedestrian",
        horizon_s=5,
        observations=(obs,),
    )
    Assessment(
        window_id=WID,
        status="timeout",
        risk_level="unknown",
        event_state="unknown",
        event_type="unknown",
        horizon_s=5,
    )
    with pytest.raises(ValidationError, match="non-ok"):
        Assessment(
            window_id=WID,
            status="failed",
            risk_level="none",
            event_state="no_supported_conflict",
            event_type="unknown",
            horizon_s=5,
        )
    with pytest.raises(ValidationError, match="observed"):
        Assessment(
            window_id=WID,
            status="ok",
            risk_level="high",
            event_state="potential_conflict",
            event_type="vehicle_pedestrian",
            horizon_s=5,
        )
    inferred = Observation(text="Might be a conflict.", kind="inferred")
    with pytest.raises(ValidationError, match="observed"):
        Assessment(
            window_id=WID,
            status="ok",
            risk_level="low",
            event_state="observed_incident",
            event_type="other",
            horizon_s=5,
            observations=(inferred,),
        )
    with pytest.raises(ValidationError, match="inconsistent"):
        Assessment(
            window_id=WID,
            status="ok",
            risk_level="high",
            event_state="no_supported_conflict",
            event_type="other",
            horizon_s=5,
            observations=(obs,),
        )
    with pytest.raises(ValidationError):
        Assessment(
            window_id=WID,
            status="ok",
            risk_level="low",
            event_state="potential_conflict",
            event_type="other",
            horizon_s=4,
        )


def test_observation_rules() -> None:
    with pytest.raises(ValidationError, match="reference"):
        Observation(text="Car brakes.", kind="observed")
    with pytest.raises(ValidationError):
        Observation(text="x" * 501, frame_ids=("f-1.000000",), kind="observed")
    with pytest.raises(ValidationError):
        Observation(text="   ", kind="limitation")


def test_failed_specialist_cannot_carry_evidence() -> None:
    obs = Observation(text="Car brakes.", frame_ids=("f-1.000000",), kind="observed")
    SpecialistEvidence(
        specialist_id="motion", window_id=WID, status="ok", observations=(obs,), uncertainty="low"
    )
    for status in ("failed", "timeout", "budget_exhausted"):
        with pytest.raises(ValidationError, match="cannot carry"):
            SpecialistEvidence(
                specialist_id="motion", window_id=WID, status=status, observations=(obs,)
            )


def test_unknown_frame_refs() -> None:
    window = synthetic_window()
    good = Observation(text="ok", frame_ids=("f-5.000000",), kind="observed")
    bad = Observation(text="future", frame_ids=("f-6.000000",), kind="observed")
    assert unknown_frame_refs([good], window) == []
    assert unknown_frame_refs([good, bad], window) == ["f-6.000000"]


def test_failed_assessment_cannot_carry_claims() -> None:
    obs = Observation(text="Car brakes.", frame_ids=("f-1.000000",), kind="observed")
    common = {"window_id": WID, "risk_level": "unknown", "event_state": "unknown", "horizon_s": 5}
    for status in ("failed", "timeout", "budget_exhausted"):
        with pytest.raises(ValidationError, match="cannot carry"):
            Assessment(status=status, event_type="unknown", observations=(obs,), **common)
        with pytest.raises(ValidationError, match="cannot carry"):
            Assessment(status=status, event_type="vehicle_pedestrian", **common)
        with pytest.raises(ValidationError, match="cannot carry"):
            Assessment(status=status, event_type="unknown", uncertainty="low", **common)
    # `unknown` (insufficient evidence) may still explain its limitations.
    limitation = Observation(text="View occluded.", kind="limitation")
    Assessment(status="unknown", event_type="unknown", observations=(limitation,), **common)
    with pytest.raises(ValidationError):
        Assessment(status="unknown", event_type="unknown", specialist_result_ids=("",), **common)
