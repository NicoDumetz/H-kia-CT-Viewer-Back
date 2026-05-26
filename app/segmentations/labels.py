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
from pathlib import Path

from app.core.config import Settings


def load_label_names(settings: Settings) -> dict[int, str]:
    labels_path = Path(settings.ct_anatomy_labels_path)

    if not settings.ct_anatomy_labels_path:
        return {}

    if not labels_path.is_file():
        return {}

    try:
        content = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return normalize_label_names(content)


def normalize_label_names(content: dict[str, str]) -> dict[int, str]:
    labels: dict[int, str] = {}

    for key, value in content.items():
        try:
            labels[int(key)] = str(value)
        except (TypeError, ValueError):
            continue

    return labels


def get_label_name(label_id: int, label_names: dict[int, str]) -> str:
    return label_names.get(label_id, f"label_{label_id}")
