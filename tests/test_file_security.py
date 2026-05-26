def test_serves_valid_study_file(client, import_nifti):
    study = import_nifti()
    response = client.get(f"/studies/{study['id']}/files/source/test1_series.nii.gz")

    assert response.status_code == 200


def test_blocks_encoded_path_traversal(client, import_nifti):
    study = import_nifti()
    response = client.get(f"/studies/{study['id']}/files/source/%2E%2E/manifest.json")

    assert response.status_code == 404


def test_blocks_plain_path_traversal(client, import_nifti):
    study = import_nifti()
    response = client.get(f"/studies/{study['id']}/files/../manifest.json")

    assert response.status_code == 404


def test_missing_study_file_returns_404(client, import_nifti):
    study = import_nifti()
    response = client.get(f"/studies/{study['id']}/files/source/missing.nii.gz")

    assert response.status_code == 404
