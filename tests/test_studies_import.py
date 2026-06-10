import asyncio
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile

from app.studies.service import prepare_dicom_source_files
from app.studies.storage import save_upload_files


def test_import_nifti_creates_manifest(client, import_nifti):
    study = import_nifti()
    manifest_path = Path(client.test_settings.storage_root) / study["id"] / "manifest.json"

    assert study["input_type"] == "nifti"
    assert study["status"] == "imported"
    assert study["files_count"] == 1
    assert manifest_path.is_file()


def test_list_and_get_imported_study(client, import_nifti):
    study = import_nifti()

    list_response = client.get("/studies")
    get_response = client.get(f"/studies/{study['id']}")

    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["id"] == study["id"]
    assert get_response.status_code == 200
    assert get_response.json()["id"] == study["id"]


def test_get_missing_study_returns_404(client):
    response = client.get("/studies/missing-study")

    assert response.status_code == 404


def test_save_upload_files_preserves_relative_paths(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    uploads = [
        build_upload_file("DICOMDIR", b"dicomdir"),
        build_upload_file("PATIENT/STUDY/SERIES/IMG0001", b"slice-1"),
        build_upload_file("PATIENT/STUDY/SERIES/IMG0001", b"slice-1-duplicate"),
    ]

    saved_paths = asyncio.run(save_upload_files(uploads, source_dir))

    assert [path.relative_to(source_dir).as_posix() for path in saved_paths] == [
        "DICOMDIR",
        "PATIENT/STUDY/SERIES/IMG0001",
        "PATIENT/STUDY/SERIES/IMG0001-1",
    ]
    assert (source_dir / "PATIENT/STUDY/SERIES/IMG0001").read_bytes() == b"slice-1"
    assert (
        source_dir / "PATIENT/STUDY/SERIES/IMG0001-1"
    ).read_bytes() == b"slice-1-duplicate"


def test_save_upload_files_sanitizes_unsafe_relative_paths(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    uploads = [build_upload_file("../evil.dcm", b"safe")]

    saved_paths = asyncio.run(save_upload_files(uploads, source_dir))

    assert saved_paths == [source_dir / "evil.dcm"]
    assert (source_dir / "evil.dcm").read_bytes() == b"safe"
    assert not (tmp_path / "evil.dcm").exists()


def test_prepare_dicom_source_files_preserves_upload_tree(tmp_path):
    upload_root = tmp_path / "original_upload"
    dicom_dir = tmp_path / "dicom"
    nested_path = upload_root / "PATIENT/STUDY/SERIES/IMG0001"
    dicomdir_path = upload_root / "DICOMDIR"
    nested_path.parent.mkdir(parents=True)
    dicom_dir.mkdir()
    nested_path.write_bytes(b"slice")
    dicomdir_path.write_bytes(b"dicomdir")

    dicom_paths = prepare_dicom_source_files(
        [dicomdir_path, nested_path],
        upload_root,
        dicom_dir,
    )

    assert [path.relative_to(dicom_dir).as_posix() for path in dicom_paths] == [
        "DICOMDIR",
        "PATIENT/STUDY/SERIES/IMG0001",
    ]
    assert (dicom_dir / "DICOMDIR").read_bytes() == b"dicomdir"
    assert (dicom_dir / "PATIENT/STUDY/SERIES/IMG0001").read_bytes() == b"slice"


def build_upload_file(filename: str, content: bytes) -> UploadFile:
    return UploadFile(BytesIO(content), filename=filename)
