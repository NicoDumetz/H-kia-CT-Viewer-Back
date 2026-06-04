import json
from pathlib import Path


def test_upload_manual_segmentation_valid(
    client,
    prepared_study,
    upload_manual_segmentation,
):
    study_id = prepared_study["study"]["id"]
    segmentation = upload_manual_segmentation(study_id)
    segmentation_dir = (
        Path(client.test_settings.storage_root)
        / study_id
        / "derived/segmentations"
        / segmentation["id"]
    )
    list_response = client.get(f"/studies/{study_id}/segmentations")
    get_response = client.get(f"/studies/{study_id}/segmentations/{segmentation['id']}")
    file_response = client.get(
        f"/studies/{study_id}/files/{segmentation['file']['relative_path']}"
    )

    assert segmentation["source_run_id"] is None
    assert segmentation["module_id"] == "manual_upload"
    assert segmentation["module_name"] == "Manual segmentation upload"
    assert segmentation["metadata"]["labels_count"] == 2
    assert (segmentation_dir / "segmentation.nii.gz").is_file()
    assert (segmentation_dir / "metadata.json").is_file()
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["id"] == segmentation["id"]
    assert get_response.status_code == 200
    assert get_response.json()["id"] == segmentation["id"]
    assert file_response.status_code == 200


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


def test_upload_manual_segmentation_shape_mismatch_returns_422(
    client,
    prepared_study,
    mismatched_segmentation_path,
):
    study_id = prepared_study["study"]["id"]

    with mismatched_segmentation_path.open("rb") as file:
        response = client.post(
            f"/studies/{study_id}/segmentations/upload",
            files={"file": (mismatched_segmentation_path.name, file, "application/gzip")},
        )

    assert response.status_code == 422


def test_upload_manual_segmentation_non_nifti_returns_422(
    client,
    prepared_study,
    tmp_path,
):
    study_id = prepared_study["study"]["id"]
    path = tmp_path / "mask.txt"
    path.write_text("not a nifti", encoding="utf-8")

    with path.open("rb") as file:
        response = client.post(
            f"/studies/{study_id}/segmentations/upload",
            files={"file": (path.name, file, "text/plain")},
        )

    assert response.status_code == 422


def test_upload_manual_segmentation_missing_study_returns_404(
    client,
    synthetic_segmentation_path,
):
    with synthetic_segmentation_path.open("rb") as file:
        response = client.post(
            "/studies/missing-study/segmentations/upload",
            files={"file": (synthetic_segmentation_path.name, file, "application/gzip")},
        )

    assert response.status_code == 404


def test_upload_manual_segmentation_without_prepared_volume_returns_404(
    client,
    import_nifti,
    synthetic_segmentation_path,
):
    study = import_nifti()

    with synthetic_segmentation_path.open("rb") as file:
        response = client.post(
            f"/studies/{study['id']}/segmentations/upload",
            files={"file": (synthetic_segmentation_path.name, file, "application/gzip")},
        )

    assert response.status_code == 404


def test_hu_analysis_after_manual_segmentation_upload(
    client,
    prepared_study,
    upload_manual_segmentation,
    create_hu_analysis,
):
    study_id = prepared_study["study"]["id"]
    segmentation = upload_manual_segmentation(study_id)
    analysis = create_hu_analysis(study_id, segmentation["id"], label_ids=[1])
    result_response = client.get(f"/studies/{study_id}/analyses/{analysis['id']}/result")
    result = result_response.json()

    assert result_response.status_code == 200
    assert result["labels_count"] == 1
    assert result["labels"][0]["label_id"] == 1
    assert result["labels"][0]["voxel_count"] > 0
