from pathlib import Path


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


def test_prepare_unknown_input_returns_422(client, import_unknown):
    study = import_unknown()
    response = client.post(f"/studies/{study['id']}/prepare")

    assert response.status_code == 422
