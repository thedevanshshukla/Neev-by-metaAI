"""Unit tests for Neev Memory Guard and RAM safety fallback."""

import pytest
from neev.app.memory_guard import (
    InsufficientRAMError,
    check_ram_headroom,
    force_cleanup,
    get_memory_stats,
)
from neev.app.runner import _build_ram_fallback_record
from neev.app.store import InspectionStore


def test_get_memory_stats():
    stats = get_memory_stats()
    assert "process_rss_mb" in stats
    assert "system_avail_mb" in stats
    assert "system_total_mb" in stats
    assert stats["process_rss_mb"] >= 0.0


def test_check_ram_headroom_passes_when_plenty():
    # Large ceiling should pass without raising
    check_ram_headroom(estimated_need_mb=10.0, peak_limit_mb=999999.0, min_system_avail_mb=1.0)


def test_check_ram_headroom_raises_when_limit_exceeded():
    # Tiny peak ceiling of 1 MB must raise InsufficientRAMError
    with pytest.raises(InsufficientRAMError) as exc_info:
        check_ram_headroom(estimated_need_mb=500.0, peak_limit_mb=1.0, min_system_avail_mb=1.0)
    assert "Peak RAM limit reached" in str(exc_info.value)


def test_build_ram_fallback_record(tmp_path):
    store = InspectionStore(tmp_path)
    # Create fake inspection
    rec = store.create(
        b"fakebytes",
        extension="png",
        image_name="test.png",
        sha256="abc123sha",
        mime="image/png",
        width=100,
        height=100,
        quality_warnings=[],
        quality_metrics={},
    )
    
    record = _build_ram_fallback_record(
        store,
        rec["inspection_id"],
        architecture="hybrid",
        error_detail="Simulated memory exhaustion"
    )
    
    assert record["status"] == "insufficient_ram"
    assert "Not enough RAM available" in record["error"]
    assert record["audit"]["inference_status"] == "insufficient_ram"
    assert "Simulated memory exhaustion" in record["audit"]["fallback_reason"]
