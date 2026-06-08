from pathlib import Path

import nibabel as nib
import numpy as np


def test_prepare_nifti_volume_creates_volume_and_metadata(client, import_nifti):
    study = import_nifti()
    study_dir = Path(client.test_settings.storage_root) / study["id"]

    before_response = client.get(f"/studies/{study['id']}/volume")
    prepare_response = client.post(f"/studies/{study['id']}/prepare")
    volume_response = client.get(f"/studies/{study['id']}/volume")
    volume_path = study_dir / "derived/volume/ct.nii.gz"
    metadata_path = study_dir / "derived/volume/metadata.json"
    body = volume_response.json()

    assert before_response.status_code == 404
    assert prepare_response.status_code == 200
    assert volume_path.is_file()
    assert metadata_path.is_file()
    assert volume_response.status_code == 200
    assert body["status"] == "prepared"
    assert body["volume"]["metadata"]["shape"] == [8, 8, 8]
    assert "intensity" in body["volume"]["metadata"]


def test_upload_nifti_endpoint_prepares_volume_and_api_alias(client, synthetic_nifti_path):
    with synthetic_nifti_path.open("rb") as file:
        response = client.post(
            "/api/studies/upload-nifti",
            files={"file": (synthetic_nifti_path.name, file, "application/gzip")},
        )

    body = response.json()
    study_dir = Path(client.test_settings.storage_root) / body["study_id"]

    assert response.status_code == 200
    assert body["status"] == "prepared"
    assert (study_dir / "source/original_upload/test1_series.nii.gz").is_file()
    assert (study_dir / "derived/volume/ct.nii.gz").is_file()
    assert body["volume"]["metadata"]["source_type"] == "nifti"


def test_upload_nifti_rejects_non_3d_volume(client, tmp_path):
    path = tmp_path / "4d.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 4, 4, 2), dtype=np.int16), np.eye(4))
    nib.save(image, str(path))

    with path.open("rb") as file:
        response = client.post(
            "/studies/upload-nifti",
            files={"file": (path.name, file, "application/gzip")},
        )

    assert response.status_code == 422
    assert "3D" in response.json()["detail"]


def test_prepare_unknown_input_returns_422(client, import_unknown):
    study = import_unknown()
    response = client.post(f"/studies/{study['id']}/prepare")

    assert response.status_code == 422
