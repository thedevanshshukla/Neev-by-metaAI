"""Cost-Controlled Live SAM 3.1 Benchmark Runner for CUBIT-Seg.

Evaluates Meta SAM 3.1 through Meta Model API on CUBIT-Seg with:
- Exact historical prompt: "crack"
- Atomic RunStore integration under runs/<run_id>/
- Persistent CUBIT cache (data/cubit_sam31_cache.json)
- Exponential backoff with jitter for HTTP 429/5xx transient failures
- Safe resume and strict duplicate prevention
- Failure preservation (failed requests are NEVER treated as empty predictions)
- Strict bipartite matching (cardinality_first, cov50)
- Decoupled detection metrics, mask metrics (IoU, Dice, precision, recall), and negative controls
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neev.adapters.base import AdapterError, BoundingBox, Detection, UrllibTransport
from neev.adapters.sam31 import (
    PRICE_PER_IMAGE_USD,
    SAM31Adapter,
    SAM31Result,
    build_image_request,
    parse_output_text,
    sanitized_request,
    sniff_image_mime,
)
from neev.config import auth_headers, load_config
from neev.eval.bipartite_matching import (
    DetectionEvaluationResult,
    evaluate_detection_validity_bipartite,
)
from neev.eval.matching import rotated_iou
from neev.measurements.masks import decode_mask, place_box_mask
from neev.persistence.runs import RunMeta, RunStore, sha256_bytes, sha256_json

CACHE_PATH = ROOT / "data" / "cubit_sam31_cache.json"
MANIFEST_PATH = ROOT / "CUBIT_DATASET_MANIFEST.json"


class ResilientSAMClient:
    """Wrapper around SAM31Adapter providing exponential backoff, rate limit handling, and telemetry."""

    def __init__(self, config, max_retries: int = 5, initial_backoff_s: float = 2.0):
        self.config = config
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s
        self.adapter = SAM31Adapter(config, UrllibTransport())
        self.call_stats = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "rate_limits_hit": 0,
            "total_latency_s": 0.0,
        }

    def infer_with_backoff(
        self, raw_bytes: bytes, phrase: str, mime: str, image_id: str
    ) -> Tuple[SAM31Result, float, int]:
        """Execute segment_image with exponential backoff and jitter."""
        retries = 0
        backoff = self.initial_backoff_s

        while True:
            self.call_stats["attempts"] += 1
            t0 = time.perf_counter()
            try:
                result = self.adapter.segment_image(raw_bytes, phrase, mime, timeout=120.0)
                latency = time.perf_counter() - t0
                self.call_stats["successes"] += 1
                self.call_stats["total_latency_s"] += latency
                return result, latency, retries
            except Exception as exc:
                latency = time.perf_counter() - t0
                err_str = str(exc)
                is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
                is_server_err = any(code in err_str for code in ("500", "502", "503", "504")) or "timed out" in err_str.lower()

                if is_rate_limit:
                    self.call_stats["rate_limits_hit"] += 1
                
                if (is_rate_limit or is_server_err) and retries < self.max_retries:
                    retries += 1
                    self.call_stats["retries"] += 1
                    sleep_time = backoff + random.uniform(0.5, 1.5)
                    print(f"[{image_id}] Transient failure ({err_str[:60]}...); retry {retries}/{self.max_retries} in {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    backoff *= 2.0
                else:
                    self.call_stats["failures"] += 1
                    raise


class LiveSAMBenchmarkRunner:
    """Manages cached inference, execution, and bipartite evaluation on CUBIT-Seg."""

    def __init__(self, runs_dir: str = "runs", cache_path: Path = CACHE_PATH):
        self.config = load_config()
        self.store = RunStore(ROOT / runs_dir)
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.client = ResilientSAMClient(self.config)

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)
        tmp.replace(self.cache_path)

    def run_image(
        self,
        image_record: dict[str, Any],
        phrase: str = "crack",
        force_live: bool = False,
    ) -> dict[str, Any]:
        """Fetch or execute SAM 3.1 inference for a single CUBIT-Seg image."""
        img_id = image_record["image_id"]
        img_path = ROOT / image_record["image_path"]
        raw_bytes = img_path.read_bytes()
        mime = sniff_image_mime(raw_bytes) or "image/jpeg"
        img_sha = sha256_bytes(raw_bytes)
        w, h = image_record["width"], image_record["height"]

        # Cache key based on stable image SHA, model, phrase, and resolution
        cache_key = f"{img_sha}_{self.config.sam_model}_{phrase}_{w}x{h}"

        # 1. Check local persistent CUBIT cache
        if not force_live and cache_key in self.cache:
            entry = self.cache[cache_key]
            if entry.get("status") == "SUCCESS":
                return {**entry, "cache_hit": True}

        # 2. Check RunStore
        img_b64 = base64.b64encode(raw_bytes).decode("ascii")
        req_payload = build_image_request(phrase, img_b64, mime)
        trace_req = sanitized_request(req_payload, img_sha)
        existing_run_id = self.store.find_cached(trace_req)

        if not force_live and existing_run_id:
            try:
                run_data = self.store.load(existing_run_id)
                out_text = run_data["response"]["output"][0]["content"][0]["text"]
                dets = parse_output_text(out_text, phrase)
                entry = self._build_cache_entry(
                    image_id=img_id,
                    image_sha=img_sha,
                    phrase=phrase,
                    run_id=existing_run_id,
                    request_id=run_data["response"].get("id", f"cached_{existing_run_id}"),
                    usage=run_data["response"].get("usage", {}),
                    raw_output_text=out_text,
                    detections=dets,
                    latency_s=0.0,
                    retries=0,
                    status="SUCCESS",
                )
                self.cache[cache_key] = entry
                self._save_cache()
                return {**entry, "cache_hit": True}
            except Exception:
                pass

        # 3. Live API Execution
        # Verify API key presence
        auth_headers(self.config)

        try:
            sam_res, latency, retries = self.client.infer_with_backoff(
                raw_bytes, phrase, mime, img_id
            )
            run_id = self.store.save(
                trace_req,
                sam_res.raw_response,
                RunMeta(
                    purpose="sam31-cubit-benchmark",
                    models={"segment": self.config.sam_model},
                    input_hashes={"image": img_sha},
                    notes={
                        "phrase": phrase,
                        "image_id": img_id,
                        "mime": mime,
                        "detections": len(sam_res.detections),
                    },
                ),
            )
            req_id = sam_res.raw_response.get("id", f"run_{run_id}")
            usage = sam_res.raw_response.get("usage", {"images_processed": 1})

            entry = self._build_cache_entry(
                image_id=img_id,
                image_sha=img_sha,
                phrase=phrase,
                run_id=run_id,
                request_id=req_id,
                usage=usage,
                raw_output_text=sam_res.raw_output_text,
                detections=sam_res.detections,
                latency_s=round(latency, 3),
                retries=retries,
                status="SUCCESS",
            )
            self.cache[cache_key] = entry
            self._save_cache()
            return {**entry, "cache_hit": False}
        except Exception as exc:
            entry = {
                "image_id": img_id,
                "image_sha256": img_sha,
                "model": self.config.sam_model,
                "phrase": phrase,
                "resolution": [w, h],
                "status": "FAILED",
                "error": str(exc),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "cache_hit": False,
            }
            # Record failed run so we never treat failure as empty predictions!
            self.cache[cache_key] = entry
            self._save_cache()
            raise

    def _build_cache_entry(
        self,
        image_id: str,
        image_sha: str,
        phrase: str,
        run_id: str,
        request_id: str,
        usage: dict[str, Any],
        raw_output_text: str,
        detections: list[Detection],
        latency_s: float,
        retries: int,
        status: str,
    ) -> dict[str, Any]:
        dets_json = []
        for d in detections:
            dets_json.append({
                "x0": d.box.x0,
                "y0": d.box.y0,
                "x1": d.box.x1,
                "y1": d.box.y1,
                "track_id": d.track_id,
                "confidence": d.confidence,
                "has_mask": bool(d.mask_payload),
                "mask_width": d.mask_width,
                "mask_height": d.mask_height,
                "mask_payload": d.mask_payload,
            })

        return {
            "image_id": image_id,
            "image_sha256": image_sha,
            "model": self.config.sam_model,
            "phrase": phrase,
            "run_id": run_id,
            "request_id": request_id,
            "usage": usage,
            "latency_s": latency_s,
            "retries": retries,
            "status": status,
            "detections_count": len(detections),
            "detections": dets_json,
            "raw_output_text": raw_output_text,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }


def decode_full_masks(
    detections: list[dict[str, Any]], width: int, height: int
) -> list[np.ndarray]:
    """Decodes all SAM mask payloads into full-canvas boolean numpy arrays."""
    masks = []
    for d in detections:
        payload = d.get("mask_payload")
        mw = d.get("mask_width")
        mh = d.get("mask_height")
        if payload and mw and mh:
            try:
                raw_raster = decode_mask(payload, mw, mh)
                b_dict = {"x0": d["x0"], "y0": d["y0"], "x1": d["x1"], "y1": d["y1"]}
                full_bytes = place_box_mask(raw_raster, mw, mh, b_dict, width, height)
                arr = np.frombuffer(full_bytes, dtype=np.uint8).reshape((height, width)).astype(bool)
                masks.append(arr)
            except Exception:
                masks.append(np.zeros((height, width), dtype=bool))
        else:
            # Fall back to bounding box fill if mask is missing
            b_mask = np.zeros((height, width), dtype=bool)
            b_mask[d["y0"]:d["y1"], d["x0"]:d["x1"]] = True
            masks.append(b_mask)
    return masks


def compute_pixel_mask_metrics(
    pred_masks: list[np.ndarray], gt_mask_path: Path, width: int, height: int
) -> dict[str, float]:
    """Compute exact pixel-level Mask IoU, Dice, Precision, and Recall against ground truth mask."""
    if pred_masks:
        pred_canvas = np.bitwise_or.reduce(pred_masks)
    else:
        pred_canvas = np.zeros((height, width), dtype=bool)

    gt_mat = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mat is None:
        gt_canvas = np.zeros((height, width), dtype=bool)
    else:
        gt_canvas = (gt_mat > 0)

    inter = int(np.count_nonzero(pred_canvas & gt_canvas))
    union = int(np.count_nonzero(pred_canvas | gt_canvas))
    pred_sum = int(np.count_nonzero(pred_canvas))
    gt_sum = int(np.count_nonzero(gt_canvas))

    if union == 0:
        return {"mask_iou": 1.0, "mask_precision": 1.0, "mask_recall": 1.0, "mask_dice": 1.0}
    if gt_sum == 0:
        return {"mask_iou": 0.0, "mask_precision": 0.0, "mask_recall": 1.0, "mask_dice": 0.0}

    iou = inter / max(1, union)
    prec = inter / max(1, pred_sum)
    rec = inter / max(1, gt_sum)
    dice = (2.0 * inter) / max(1, pred_sum + gt_sum)

    return {
        "mask_iou": round(float(iou), 4),
        "mask_precision": round(float(prec), 4),
        "mask_recall": round(float(rec), 4),
        "mask_dice": round(float(dice), 4),
    }


def evaluate_sam_image(
    image_record: dict[str, Any],
    sam_result: dict[str, Any],
    matching_mode: str = "cardinality_first",
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Run strict threshold-constrained bipartite matching and compute all detection + mask metrics."""
    w, h = image_record["width"], image_record["height"]
    gt_instances = image_record.get("instances", [])
    n_gts = len(gt_instances)

    gt_rasters = []
    gt_polys = []
    for inst in gt_instances:
        m = Image.new("1", (w, h), 0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(m)
        pts = [tuple(p) for p in inst["polygon"]]
        if len(pts) >= 3:
            d.polygon([(int(pt[0]), int(pt[1])) for pt in pts], fill=1)
        gt_rasters.append(np.array(m, dtype=bool))
        gt_polys.append(pts)

    pred_boxes = [
        {"x0": d["x0"], "y0": d["y0"], "x1": d["x1"], "y1": d["y1"]}
        for d in sam_result.get("detections", [])
    ]
    pred_masks = decode_full_masks(sam_result.get("detections", []), w, h)

    eval_res = evaluate_detection_validity_bipartite(
        prediction_boxes=pred_boxes,
        gt_rasters=gt_rasters,
        gt_polygons=gt_polys,
        min_coverage_thresh=threshold,
        pred_masks=pred_masks,
        matching_mode=matching_mode,
        image_width=w,
        image_height=h,
    )

    mask_metrics = compute_pixel_mask_metrics(
        pred_masks, ROOT / image_record["mask_path"], w, h
    )

    # Rotated box IoU calculation
    rot_ious = []
    for pb in pred_boxes:
        poly_p = [
            (float(pb["x0"]), float(pb["y0"])),
            (float(pb["x1"]), float(pb["y0"])),
            (float(pb["x1"]), float(pb["y1"])),
            (float(pb["x0"]), float(pb["y1"])),
        ]
        best_r = 0.0
        for gp in gt_polys:
            if len(gp) >= 3:
                r_iou = rotated_iou(poly_p, gp)
                if r_iou > best_r:
                    best_r = r_iou
        rot_ious.append(best_r)
    mean_rot_iou = float(np.mean(rot_ious)) if rot_ious else 0.0

    return {
        "image_id": image_record["image_id"],
        "split": image_record["split"],
        "has_crack": image_record["has_crack"],
        "pilot_category": image_record.get("pilot_category", "general"),
        "width_category": image_record.get("width_category", "unknown"),
        "is_branching": image_record.get("is_branching", False),
        "mean_crack_width_px": image_record.get("mean_crack_width_px", 0.0),
        "gt_instances_count": n_gts,
        "pred_count": len(pred_boxes),
        "true_positives": eval_res.true_positives,
        "false_positives": eval_res.false_positives,
        "false_negatives": eval_res.false_negatives,
        "precision": round(eval_res.precision, 4),
        "recall": round(eval_res.recall, 4),
        "f1": round(eval_res.f1, 4),
        "unconstrained_coverage_pct": eval_res.unconstrained_coverage_pct,
        "mask_iou": mask_metrics["mask_iou"],
        "mask_precision": mask_metrics["mask_precision"],
        "mask_recall": mask_metrics["mask_recall"],
        "mask_dice": mask_metrics["mask_dice"],
        "mean_rotated_box_iou": round(mean_rot_iou, 4),
        "latency_s": sam_result.get("latency_s", 0.0),
        "request_id": sam_result.get("request_id"),
        "run_id": sam_result.get("run_id"),
        "cache_hit": sam_result.get("cache_hit", False),
    }
