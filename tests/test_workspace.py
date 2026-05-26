def test_workspace_after_import_before_prepare(client, import_nifti):
    study = import_nifti()
    response = client.get(f"/studies/{study['id']}/workspace")
    body = response.json()

    assert response.status_code == 200
    assert body["volume"]["is_prepared"] is False
    assert body["available_actions"]["can_prepare_volume"] is True
    assert body["available_actions"]["can_create_ai_run"] is False


def test_workspace_after_prepare(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.get(f"/studies/{study_id}/workspace")
    body = response.json()

    assert response.status_code == 200
    assert body["volume"]["is_prepared"] is True
    assert body["available_actions"]["can_create_ai_run"] is True


def test_workspace_after_segmentation_and_analysis(
    client,
    published_segmentation,
    create_hu_analysis,
):
    study_id = published_segmentation["study_id"]
    segmentation_id = published_segmentation["segmentation"]["id"]
    create_hu_analysis(study_id, segmentation_id)
    response = client.get(f"/studies/{study_id}/workspace")
    body = response.json()

    assert response.status_code == 200
    assert body["ai"]["runs"]
    assert body["segmentations"]["items"]
    assert body["segmentations"]["latest"] is not None
    assert body["analyses"]["items"]
    assert body["analyses"]["latest"] is not None
    assert body["available_actions"]["can_run_label_hu_statistics"] is True


def test_workspace_missing_study_returns_404(client):
    response = client.get("/studies/missing-study/workspace")

    assert response.status_code == 404
