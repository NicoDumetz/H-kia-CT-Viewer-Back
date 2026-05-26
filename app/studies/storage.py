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
from uuid import uuid4

from fastapi import UploadFile


def create_study_id() -> str:
    return str(uuid4())


def create_source_dir(storage_root: str, study_id: str) -> Path:
    source_dir = Path(storage_root) / study_id / "source"
    source_dir.mkdir(parents=True, exist_ok=False)
    return source_dir


def get_study_dir(storage_root: str, study_id: str) -> Path:
    return Path(storage_root) / study_id


def get_source_dir(storage_root: str, study_id: str) -> Path:
    return get_study_dir(storage_root, study_id) / "source"


def resolve_study_file(storage_root: str, study_id: str, relative_path: str) -> Path | None:
    study_dir = get_study_dir(storage_root, study_id).resolve()
    requested_path = Path(relative_path)

    if requested_path.is_absolute():
        return None

    if ".." in requested_path.parts:
        return None

    file_path = (study_dir / requested_path).resolve()

    if not file_path.is_relative_to(study_dir):
        return None

    if not file_path.is_file():
        return None

    return file_path


def list_study_dirs(storage_root: str) -> list[Path]:
    root = Path(storage_root)

    if not root.is_dir():
        return []

    return [path for path in root.iterdir() if path.is_dir()]


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    return name or "upload.bin"


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    chunk_size = 1024 * 1024
    chunk = await upload_file.read(chunk_size)

    with destination.open("wb") as output:
        while chunk:
            output.write(chunk)
            chunk = await upload_file.read(chunk_size)

    await upload_file.seek(0)


async def save_upload_files(files: list[UploadFile], source_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    used_names: set[str] = set()

    for index, upload_file in enumerate(files, start=1):
        filename = safe_filename(upload_file.filename or f"upload-{index}.bin")
        destination = unique_destination(source_dir, filename, used_names)

        await save_upload_file(upload_file, destination)
        saved_paths.append(destination)

    return saved_paths


def unique_destination(source_dir: Path, filename: str, used_names: set[str]) -> Path:
    path = source_dir / filename
    stem = path.stem
    suffix = path.suffix
    counter = 1

    while path.name in used_names or path.exists():
        path = source_dir / f"{stem}-{counter}{suffix}"
        counter += 1

    used_names.add(path.name)
    return path
