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
