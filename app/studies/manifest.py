# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : manifest.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from pathlib import Path

from app.studies.schemas import StudyFileRead, StudyListItem, StudyRead


MANIFEST_FILENAME = "manifest.json"


def build_source_files(study_dir: Path, paths: list[Path]) -> list[StudyFileRead]:
    source_files: list[StudyFileRead] = []

    for path in paths:
        source_files.append(
            StudyFileRead(
                filename=path.name,
                relative_path=path.relative_to(study_dir).as_posix(),
                size_bytes=path.stat().st_size,
            )
        )

    return source_files


def write_manifest(study_dir: Path, study: StudyRead) -> Path:
    manifest_path = study_dir / MANIFEST_FILENAME
    content = study.model_dump_json(indent=2)

    manifest_path.write_text(content, encoding="utf-8")
    return manifest_path


def read_manifest(study_dir: Path) -> StudyRead | None:
    manifest_path = study_dir / MANIFEST_FILENAME

    if not manifest_path.is_file():
        return None

    return StudyRead.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def to_list_item(study: StudyRead) -> StudyListItem:
    return StudyListItem(
        id=study.id,
        status=study.status,
        input_type=study.input_type,
        files_count=study.files_count,
        metadata=study.metadata,
        created_at=study.created_at,
        updated_at=study.updated_at,
        error=study.error,
    )
