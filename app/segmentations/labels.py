# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : labels.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

import json
from colorsys import hsv_to_rgb
from pathlib import Path
from typing import Any

from app.core.config import Settings


TOTALSEG117_MODEL_ID = "totalseg117_3d_lowres"

TOTALSEG117_LABELS: dict[int, str] = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages",
}


def load_label_names(settings: Settings) -> dict[int, str]:
    labels_path = Path(settings.ct_anatomy_labels_path)

    if not settings.ct_anatomy_labels_path:
        return dict(TOTALSEG117_LABELS)

    if not labels_path.is_file():
        return dict(TOTALSEG117_LABELS)

    try:
        content = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return dict(TOTALSEG117_LABELS)

    labels = dict(TOTALSEG117_LABELS)
    labels.update(normalize_label_names(content))

    return labels


def normalize_label_names(content: Any) -> dict[int, str]:
    labels: dict[int, str] = {}

    if isinstance(content, dict) and isinstance(content.get("labels"), list):
        content = {
            item.get("id", item.get("label_id")): item.get("name")
            for item in content["labels"]
            if isinstance(item, dict)
        }

    if not isinstance(content, dict):
        return labels

    for key, value in content.items():
        label_id = parse_label_id(key)

        if label_id is None or value is None:
            continue

        labels[label_id] = str(value)

    return labels


def parse_label_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_label_name(label_id: int, label_names: dict[int, str]) -> str:
    return label_names.get(label_id, f"label_{label_id}")


def get_label_group(name: str) -> str:
    if name.startswith("vertebrae_"):
        return "vertebrae"

    if name == "sacrum":
        return "sacrum"

    if name.startswith("rib_") or name in {"sternum", "costal_cartilages", "clavicula_left", "clavicula_right", "scapula_left", "scapula_right"}:
        return "thoracic_bones"

    if name in {"humerus_left", "humerus_right", "femur_left", "femur_right", "hip_left", "hip_right", "skull"}:
        return "bones"

    if name.startswith("lung_") or name == "trachea":
        return "lungs"

    if "artery" in name or "vena" in name or "vein" in name or name in {"aorta", "pulmonary_vein", "portal_vein_and_splenic_vein"}:
        return "vessels"

    if any(token in name for token in ("gluteus", "iliopsoas", "autochthon")):
        return "muscles"

    if name in {
        "spleen",
        "kidney_right",
        "kidney_left",
        "gallbladder",
        "liver",
        "stomach",
        "pancreas",
        "adrenal_gland_right",
        "adrenal_gland_left",
        "small_bowel",
        "duodenum",
        "colon",
        "esophagus",
    }:
        return "abdominal_organs"

    if name in {"urinary_bladder", "prostate"}:
        return "pelvic_organs"

    return "other"


def get_label_color(label_id: int) -> str:
    hue = ((label_id * 137.508) % 360) / 360
    red, green, blue = hsv_to_rgb(hue, 0.65, 0.9)

    return f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"
