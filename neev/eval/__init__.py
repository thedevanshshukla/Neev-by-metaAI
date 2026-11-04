"""Evaluation: labels, benchmark, matching, and variant comparison."""

from neev.eval.benchmark import (
    CATEGORIES,
    label_status,
    load_benchmark,
    validate_benchmark,
)
from neev.eval.compare import (
    PROMPT_TO_CATEGORY,
    collect_predictions,
    run_evaluation,
    run_rotated_evaluation,
)
from neev.eval.labelstudio import (
    LSImage,
    LSRect,
    load_export,
    ls_rotation_to_corners,
    parse_task,
    rect_polygon,
    resolve_image_path,
    validate_export,
)
from neev.eval.crops import (
    load_crops_config,
    materialize,
    validate_crop_box,
    validate_crops_config,
)
from neev.eval.labels import load_labels, validate_labels
from neev.eval.matching import (
    clip_polygon_to_image,
    mask_iou,
    match_predictions,
    match_rotated,
    polygon_area,
    polygon_intersection_area,
    prf,
    rle_decode,
    rotated_iou,
)
from neev.eval.metrics import agreement, binary_pr, consistency, exact_match_rate
from neev.eval.geometry_eval import (
    GeometryAwareReport,
    derive_geometry,
    evaluate_geometry_aware,
    mask_to_oriented_box,
    minimum_bounding_rectangle,
)

__all__ = [
    "CATEGORIES",
    "PROMPT_TO_CATEGORY",
    "label_status",
    "load_benchmark",
    "validate_benchmark",
    "collect_predictions",
    "run_evaluation",
    "run_rotated_evaluation",
    "LSImage",
    "LSRect",
    "load_export",
    "ls_rotation_to_corners",
    "parse_task",
    "rect_polygon",
    "resolve_image_path",
    "validate_export",
    "load_crops_config",
    "materialize",
    "validate_crop_box",
    "validate_crops_config",
    "load_labels",
    "validate_labels",
    "clip_polygon_to_image",
    "mask_iou",
    "match_predictions",
    "match_rotated",
    "polygon_area",
    "polygon_intersection_area",
    "prf",
    "rle_decode",
    "rotated_iou",
    "agreement",
    "binary_pr",
    "consistency",
    "exact_match_rate",
    "minimum_bounding_rectangle",
    "mask_to_oriented_box",
    "derive_geometry",
    "evaluate_geometry_aware",
    "GeometryAwareReport",
]
