from pathlib import Path


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
