import json
from pathlib import Path


def test_publish_segmentation_from_simulated_run(client, published_segmentation):
    study_id = published_segmentation["study_id"]
    segmentation = published_segmentation["segmentation"]
    segmentation_dir = (
        Path(client.test_settings.storage_root)
        / study_id
        / "derived/segmentations"
        / segmentation["id"]
    )
    labels = segmentation["metadata"]["labels"]
    label_ids = [label["label_id"] for label in labels]
    list_response = client.get(f"/studies/{study_id}/segmentations")
    get_response = client.get(f"/studies/{study_id}/segmentations/{segmentation['id']}")

    assert (segmentation_dir / "segmentation.nii.gz").is_file()
    assert (segmentation_dir / "metadata.json").is_file()
    assert segmentation["metadata"]["labels_count"] > 0
    assert 0 not in label_ids
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["id"] == segmentation["id"]
    assert get_response.status_code == 200
    assert get_response.json()["id"] == segmentation["id"]


def test_missing_segmentation_returns_404(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.get(f"/studies/{study_id}/segmentations/missing-segmentation")

    assert response.status_code == 404


def test_publish_pending_run_returns_422(client, prepared_study, create_ai_run):
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id)
    response = client.post(f"/studies/{study_id}/ai-runs/{run['id']}/publish-segmentation")

    assert response.status_code == 422


def test_publish_run_without_segmentation_artifact_returns_422(
    client,
    prepared_study,
    create_ai_run,
    simulate_run,
):
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id)
    simulated = simulate_run(study_id, run["id"])
    run_path = (
        Path(client.test_settings.storage_root)
        / study_id
        / "derived/ai-runs"
        / simulated["id"]
        / "run.json"
    )
    run_data = json.loads(run_path.read_text(encoding="utf-8"))
    run_data["output"]["artifacts"] = [
        artifact
        for artifact in run_data["output"]["artifacts"]
        if artifact["type"] != "nifti_segmentation"
    ]
    run_path.write_text(json.dumps(run_data), encoding="utf-8")

    response = client.post(
        f"/studies/{study_id}/ai-runs/{simulated['id']}/publish-segmentation"
    )

    assert response.status_code == 422
