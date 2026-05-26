def test_ai_modules_include_expected_modules(client):
    response = client.get("/ai/modules")
    modules = {item["id"]: item for item in response.json()["items"]}

    assert response.status_code == 200
    assert "ct_anatomy_segmentation_nnunet" in modules
    assert "segmentation_label_hu_statistics" in modules
    assert modules["segmentation_label_hu_statistics"]["is_available"] is True


def test_create_list_and_get_ai_run(client, prepared_study, create_ai_run):
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id)
    list_response = client.get(f"/studies/{study_id}/ai-runs")
    get_response = client.get(f"/studies/{study_id}/ai-runs/{run['id']}")

    assert run["status"] == "pending"
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["id"] == run["id"]
    assert get_response.status_code == 200
    assert get_response.json()["id"] == run["id"]


def test_create_ai_run_before_prepare_returns_422(client, import_nifti):
    study = import_nifti()
    response = client.post(
        f"/studies/{study['id']}/ai-runs",
        json={"module_id": "ct_anatomy_segmentation_nnunet"},
    )

    assert response.status_code == 422


def test_create_ai_run_with_unknown_module_returns_422(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.post(
        f"/studies/{study_id}/ai-runs",
        json={"module_id": "missing_module"},
    )

    assert response.status_code == 422


def test_missing_ai_run_returns_404(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.get(f"/studies/{study_id}/ai-runs/missing-run")

    assert response.status_code == 404


def test_simulate_ai_run_succeeds(client, prepared_study, create_ai_run, simulate_run):
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id)
    simulated = simulate_run(study_id, run["id"])
    artifacts = [artifact["type"] for artifact in simulated["output"]["artifacts"]]

    assert simulated["status"] == "succeeded"
    assert "nifti_segmentation" in artifacts


def test_execute_with_nnunet_disabled_returns_422_and_keeps_pending(
    client,
    prepared_study,
    create_ai_run,
):
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id)
    execute_response = client.post(f"/studies/{study_id}/ai-runs/{run['id']}/execute")
    get_response = client.get(f"/studies/{study_id}/ai-runs/{run['id']}")

    assert execute_response.status_code == 422
    assert get_response.json()["status"] == "pending"


def test_execute_non_nnunet_module_returns_422(client, prepared_study, create_ai_run):
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id, module_id="osteoporosis_hu_analysis")
    response = client.post(f"/studies/{study_id}/ai-runs/{run['id']}/execute")

    assert response.status_code == 422
