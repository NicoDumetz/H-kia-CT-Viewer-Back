# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : storage.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from pathlib import Path

from app.analyses.schemas import AnalysisRead, AnalysisResultRead
from app.studies.storage import get_study_dir


ANALYSIS_FILENAME = "analysis.json"
RESULT_FILENAME = "result.json"


def get_analyses_dir(storage_root: str, study_id: str) -> Path:
    return get_study_dir(storage_root, study_id) / "derived" / "analyses"


def create_analysis_dir(storage_root: str, study_id: str, analysis_id: str) -> Path:
    analysis_dir = get_analyses_dir(storage_root, study_id) / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=False)
    return analysis_dir


def get_analysis_dir(storage_root: str, study_id: str, analysis_id: str) -> Path:
    return get_analyses_dir(storage_root, study_id) / analysis_id


def list_analysis_dirs(storage_root: str, study_id: str) -> list[Path]:
    analyses_dir = get_analyses_dir(storage_root, study_id)

    if not analyses_dir.is_dir():
        return []

    return [path for path in analyses_dir.iterdir() if path.is_dir()]


def write_analysis(analysis_dir: Path, analysis: AnalysisRead) -> Path:
    analysis_path = analysis_dir / ANALYSIS_FILENAME

    analysis_path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    return analysis_path


def read_analysis(analysis_dir: Path) -> AnalysisRead | None:
    analysis_path = analysis_dir / ANALYSIS_FILENAME

    if not analysis_path.is_file():
        return None

    return AnalysisRead.model_validate_json(analysis_path.read_text(encoding="utf-8"))


def write_analysis_result(analysis_dir: Path, result: AnalysisResultRead) -> Path:
    result_path = analysis_dir / RESULT_FILENAME

    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result_path


def read_analysis_result(analysis_dir: Path) -> AnalysisResultRead | None:
    result_path = analysis_dir / RESULT_FILENAME

    if not result_path.is_file():
        return None

    return AnalysisResultRead.model_validate_json(result_path.read_text(encoding="utf-8"))
