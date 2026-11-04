"""Tests for offline crack DSP candidate proposal generation."""

import numpy as np
import pytest

from neev.measurements.crack_dsp import (
    CrackCandidateProposal,
    propose_crack_candidate_boxes,
)


def test_propose_empty_or_flat_image():
    # Completely flat image
    flat = np.full((100, 100), 128, dtype=np.uint8)
    props = propose_crack_candidate_boxes(flat)
    assert props == []

    # Invalid dimension
    with pytest.raises(ValueError):
        propose_crack_candidate_boxes(np.zeros((10, 10, 3), dtype=np.uint8))


def test_propose_synthetic_crack():
    # Construct an artificial dark diagonal crack line on bright wall
    img = np.full((200, 200), 220, dtype=np.uint8)
    for i in range(40, 160):
        # Jagged/tortuous diagonal crack
        offset = int(np.sin(i / 8.0) * 4)
        img[i, i + offset] = 30
        img[i, i + offset + 1] = 30

    props = propose_crack_candidate_boxes(img, max_proposals=5)
    assert len(props) >= 1

    top = props[0]
    assert isinstance(top, CrackCandidateProposal)
    assert top.box["x0"] < top.box["x1"]
    assert top.box["y0"] < top.box["y1"]
    assert 0 <= top.box["x0"] <= 200
    assert 0 <= top.box["x1"] <= 200
    assert 0 <= top.box["y0"] <= 200
    assert 0 <= top.box["y1"] <= 200

    # Ensure crack line is enclosed in proposal box
    assert top.box["x0"] <= 45
    assert top.box["y0"] <= 45
    assert top.box["x1"] >= 155
    assert top.box["y1"] >= 155

    d = top.to_dict()
    assert d["proposal_id"] == top.proposal_id
    assert d["salience_score"] > 0
    assert d["energy"] > 0
    assert d["pixel_count"] > 0
    assert d["tortuosity"] >= 1.0


def test_deterministic_proposals():
    # Verify exact determinism across multiple runs
    np.random.seed(42)
    noise_img = np.random.randint(180, 220, (150, 150), dtype=np.uint8)
    noise_img[50:100, 75] = 20  # vertical defect

    p1 = propose_crack_candidate_boxes(noise_img)
    p2 = propose_crack_candidate_boxes(noise_img)

    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        assert a.box == b.box
        assert a.salience_score == b.salience_score
        assert a.energy == b.energy
        assert a.is_rectilinear == b.is_rectilinear


def test_propose_min_energy_filter():
    # Construct an image with a high-energy crack and a low-energy faint artifact
    img = np.full((200, 200), 220, dtype=np.uint8)
    # Long strong crack
    for i in range(30, 170):
        img[i, 100] = 30
        img[i, 101] = 30
    # Short weak defect
    for i in range(50, 80):
        img[i, 40] = 130

    props_all = propose_crack_candidate_boxes(img, max_proposals=10, min_energy=0.0)
    assert len(props_all) >= 1
    energies = [p.energy for p in props_all]
    max_e = max(energies)

    # Filter with threshold higher than lowest energy
    if len(energies) > 1:
        min_cutoff = sorted(energies)[-1]  # only top energy
        props_filtered = propose_crack_candidate_boxes(img, max_proposals=10, min_energy=min_cutoff - 0.1)
        assert len(props_filtered) < len(props_all)
        assert all(p.energy >= min_cutoff - 0.1 for p in props_filtered)

