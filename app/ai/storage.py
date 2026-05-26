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

from app.ai.schemas import AiRunRead
from app.studies.storage import get_study_dir


RUN_FILENAME = "run.json"
LOGS_FILENAME = "logs.txt"
OUTPUTS_DIRNAME = "outputs"


def get_ai_runs_dir(storage_root: str, study_id: str) -> Path:
    return get_study_dir(storage_root, study_id) / "derived" / "ai-runs"


def create_ai_run_dir(storage_root: str, study_id: str, run_id: str) -> Path:
    run_dir = get_ai_runs_dir(storage_root, study_id) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_ai_run_dir(storage_root: str, study_id: str, run_id: str) -> Path:
    return get_ai_runs_dir(storage_root, study_id) / run_id


def get_outputs_dir(run_dir: Path) -> Path:
    return run_dir / OUTPUTS_DIRNAME


def get_nnunet_input_dir(run_dir: Path) -> Path:
    return run_dir / "nnunet" / "input"


def get_nnunet_output_dir(run_dir: Path) -> Path:
    return run_dir / "nnunet" / "output"


def write_ai_run(run_dir: Path, run: AiRunRead) -> Path:
    run_path = run_dir / RUN_FILENAME
    content = run.model_dump_json(indent=2)

    run_path.write_text(content, encoding="utf-8")
    return run_path


def read_ai_run(run_dir: Path) -> AiRunRead | None:
    run_path = run_dir / RUN_FILENAME

    if not run_path.is_file():
        return None

    return AiRunRead.model_validate_json(run_path.read_text(encoding="utf-8"))


def list_ai_run_dirs(storage_root: str, study_id: str) -> list[Path]:
    runs_dir = get_ai_runs_dir(storage_root, study_id)

    if not runs_dir.is_dir():
        return []

    return [path for path in runs_dir.iterdir() if path.is_dir()]
