import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from app.ai.nnunet_executor import get_backend_root


VALID_CHECKPOINT_SIZE_BYTES = 140 * 1024 * 1024
TRUNCATED_CHECKPOINT_SIZE_BYTES = 119 * 1024 * 1024


def get_nnunet_module(client):
    response = client.get("/api/ai/modules")
    modules = {item["id"]: item for item in response.json()["items"]}

    assert response.status_code == 200
    return modules["ct_anatomy_segmentation_nnunet"]


def write_nnunet_results_tree(
    tmp_path: Path,
    checkpoint_name: str = "checkpoint_final.pth",
    write_plans: bool = True,
    write_dataset_json: bool = True,
    write_checkpoint: bool = True,
    checkpoint_size_bytes: int = VALID_CHECKPOINT_SIZE_BYTES,
) -> Path:
    configuration_dir = (
        tmp_path
        / "nnUNet_results"
        / "Dataset501_TotalSegmentator117"
        / "nnUNetTrainer__nnUNetPlans__3d_lowres"
    )
    fold_dir = configuration_dir / "fold_0"

    fold_dir.mkdir(parents=True)

    if write_plans:
        (configuration_dir / "plans.json").write_text("{}", encoding="utf-8")

    if write_dataset_json:
        (configuration_dir / "dataset.json").write_text("{}", encoding="utf-8")

    if write_checkpoint:
        checkpoint_path = fold_dir / checkpoint_name

        with checkpoint_path.open("wb") as checkpoint_file:
            checkpoint_file.truncate(checkpoint_size_bytes)

    return tmp_path / "nnUNet_results"


def test_ai_modules_include_expected_modules(client):
    response = client.get("/ai/modules")
    modules = {item["id"]: item for item in response.json()["items"]}

    assert response.status_code == 200
    assert "ct_anatomy_segmentation_nnunet" in modules
    assert "availability_error" in modules["ct_anatomy_segmentation_nnunet"]
    assert "segmentation_label_hu_statistics" in modules
    assert modules["segmentation_label_hu_statistics"]["is_available"] is True
    assert modules["segmentation_label_hu_statistics"]["availability_error"] is None


def test_nnunet_module_unavailable_when_disabled(client):
    module = get_nnunet_module(client)

    assert module["is_available"] is False
    assert module["availability_error"] == "NNUNET_ENABLED=false"


def test_nnunet_module_unavailable_when_results_dir_empty(client):
    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_default_dataset = "501"
    client.test_settings.nnunet_results_dir = ""

    module = get_nnunet_module(client)

    assert module["is_available"] is False
    assert module["availability_error"] == "NNUNET_RESULTS_DIR is not configured"


def test_nnunet_module_unavailable_when_plans_missing(client, tmp_path):
    results_dir = write_nnunet_results_tree(tmp_path, write_plans=False)
    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "501"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_final.pth"

    module = get_nnunet_module(client)

    assert module["is_available"] is False
    assert module["availability_error"].startswith("Missing plans.json:")


def test_nnunet_module_unavailable_when_dataset_json_missing(client, tmp_path):
    results_dir = write_nnunet_results_tree(tmp_path, write_dataset_json=False)
    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "501"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_final.pth"

    module = get_nnunet_module(client)

    assert module["is_available"] is False
    assert module["availability_error"].startswith("Missing dataset.json:")


def test_nnunet_module_unavailable_when_configured_checkpoint_missing(client, tmp_path):
    results_dir = write_nnunet_results_tree(
        tmp_path,
        checkpoint_name="checkpoint_best.pth",
    )
    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "501"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_final.pth"

    module = get_nnunet_module(client)

    assert module["is_available"] is False
    assert module["availability_error"].startswith("Missing checkpoint:")
    assert module["availability_error"].endswith("fold_0/checkpoint_final.pth")


def test_nnunet_module_unavailable_when_checkpoint_truncated(client, tmp_path):
    results_dir = write_nnunet_results_tree(
        tmp_path,
        checkpoint_name="checkpoint_best.pth",
        checkpoint_size_bytes=TRUNCATED_CHECKPOINT_SIZE_BYTES,
    )
    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "501"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_best.pth"

    module = get_nnunet_module(client)

    assert module["is_available"] is False
    assert module["availability_error"].startswith(
        "Checkpoint file appears corrupted or unreadable:"
    )


def test_nnunet_module_available_with_configured_checkpoint_final(
    client,
    tmp_path,
    monkeypatch,
):
    results_dir = write_nnunet_results_tree(
        tmp_path,
        checkpoint_name="checkpoint_final.pth",
    )
    monkeypatch.setattr(
        "app.ai.nnunet_executor.shutil.which",
        lambda command: "/usr/bin/nnUNetv2_predict",
    )
    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "501"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_final.pth"

    module = get_nnunet_module(client)

    assert module["is_available"] is True
    assert module["availability_error"] is None


def test_nnunet_module_unavailable_when_command_missing(client, tmp_path, monkeypatch):
    results_dir = write_nnunet_results_tree(
        tmp_path,
        checkpoint_name="checkpoint_final.pth",
    )
    monkeypatch.setattr("app.ai.nnunet_executor.shutil.which", lambda command: None)
    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "501"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_final.pth"

    module = get_nnunet_module(client)

    assert module["is_available"] is False
    assert module["availability_error"] == (
        "nnU-Net predict command not found in PATH: nnUNetv2_predict"
    )


def test_nnunet_module_available_with_configured_checkpoint_best(
    client,
    tmp_path,
    monkeypatch,
):
    results_dir = write_nnunet_results_tree(
        tmp_path,
        checkpoint_name="checkpoint_best.pth",
    )
    monkeypatch.setattr(
        "app.ai.nnunet_executor.shutil.which",
        lambda command: "/usr/bin/nnUNetv2_predict",
    )
    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "Dataset501_TotalSegmentator117"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_best.pth"

    module = get_nnunet_module(client)

    assert module["is_available"] is True
    assert module["availability_error"] is None


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


def test_execute_with_incomplete_nnunet_artifacts_returns_clear_error(
    client,
    prepared_study,
    create_ai_run,
    tmp_path,
):
    results_dir = write_nnunet_results_tree(tmp_path, write_plans=False)
    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "Dataset501_TotalSegmentator117"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_final.pth"
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id)

    execute_response = client.post(f"/studies/{study_id}/ai-runs/{run['id']}/execute")
    get_response = client.get(f"/studies/{study_id}/ai-runs/{run['id']}")

    assert execute_response.status_code == 422
    assert execute_response.json()["detail"].startswith("Missing plans.json:")
    assert get_response.json()["status"] == "pending"


def test_execute_nnunet_uses_absolute_lowres_input_and_resamples_final_mask(
    client,
    prepared_study,
    create_ai_run,
    tmp_path,
    monkeypatch,
):
    results_dir = write_nnunet_results_tree(
        tmp_path,
        checkpoint_name="checkpoint_best.pth",
    )
    command_path = tmp_path / "venv" / "bin" / "nnUNetv2_predict"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "app.ai.nnunet_executor.shutil.which",
        lambda command: str(command_path),
    )

    def fake_run(
        command,
        cwd,
        capture_output,
        text,
        env,
        timeout,
        check,
    ):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env

        lowres_dir = Path(command[command.index("-i") + 1])
        output_lowres_dir = Path(command[command.index("-o") + 1])
        lowres_inputs = sorted(lowres_dir.glob("*_0000.nii.gz"))

        assert capture_output is True
        assert text is True
        assert check is False
        assert timeout == client.test_settings.nnunet_timeout_seconds
        assert Path(command[0]).is_absolute()
        assert lowres_dir.is_absolute()
        assert output_lowres_dir.is_absolute()
        assert Path(cwd) == get_backend_root()
        assert command[command.index("-device") + 1] == "cpu"
        assert "--disable_tta" in command
        assert command[command.index("-npp") + 1] == "1"
        assert command[command.index("-nps") + 1] == "1"
        assert env["nnUNet_raw"] == str(get_backend_root() / "storage" / "nnunet" / "raw")
        assert env["nnUNet_preprocessed"] == str(
            get_backend_root() / "storage" / "nnunet" / "preprocessed"
        )
        assert env["nnUNet_results"] == str(results_dir.resolve())
        assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
        assert env["PYTHONNOUSERSITE"] == "1"
        assert lowres_dir.name == "lowres"
        assert output_lowres_dir.name == "output_lowres"
        assert len(lowres_inputs) == 1

        lowres_input = lowres_inputs[0]
        case_id = lowres_input.name.removesuffix("_0000.nii.gz")
        lowres_image = nib.load(str(lowres_input))
        mask = np.zeros(lowres_image.shape, dtype=np.uint8)
        mask.flat[0] = 1

        output_lowres_dir.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(mask, lowres_image.affine),
            str(output_lowres_dir / f"{case_id}.nii.gz"),
        )

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="fake stdout",
            stderr="fake stderr",
        )

    monkeypatch.setattr("app.ai.nnunet_executor.subprocess.run", fake_run)

    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_predict_command = str(command_path)
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "501"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_best.pth"
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id)

    execute_response = client.post(f"/studies/{study_id}/ai-runs/{run['id']}/execute")

    assert execute_response.status_code == 200
    assert execute_response.json()["status"] == "succeeded"

    command = captured["command"]
    lowres_dir = Path(command[command.index("-i") + 1])
    job_dir = lowres_dir.parent
    logs_path = (
        Path(client.test_settings.storage_root)
        / study_id
        / "derived"
        / "ai-runs"
        / run["id"]
        / "logs.txt"
    )
    logs_text = logs_path.read_text(encoding="utf-8")
    segmentation_path = (
        Path(client.test_settings.storage_root)
        / study_id
        / "derived"
        / "ai-runs"
        / run["id"]
        / "outputs"
        / "segmentation.nii.gz"
    )
    original_path = (
        Path(client.test_settings.storage_root)
        / study_id
        / "derived"
        / "volume"
        / "ct.nii.gz"
    )
    final_mask = nib.load(str(segmentation_path))
    original_ct = nib.load(str(original_path))
    final_array = np.asanyarray(final_mask.dataobj)

    assert f"-i {lowres_dir}" in logs_text
    assert f"-i {job_dir / 'input'}" not in logs_text
    assert f"cwd={captured['cwd']}" in logs_text
    assert f"job_dir={job_dir}" in logs_text
    assert f"input_dir={job_dir / 'input'}" in logs_text
    assert f"lowres_dir={lowres_dir}" in logs_text
    assert f"output_lowres_dir={job_dir / 'output_lowres'}" in logs_text
    assert f"nnUNet_results={results_dir.resolve()}" in logs_text
    assert "device=cpu" in logs_text
    assert "checkpoint=checkpoint_best.pth" in logs_text
    assert f"checkpoint_path={results_dir.resolve()}" in logs_text
    assert "stdout:\nfake stdout\n" in logs_text
    assert "stderr:\nfake stderr\n" in logs_text
    assert (job_dir / "input").is_dir()
    assert (job_dir / "lowres").is_dir()
    assert (job_dir / "output_lowres").is_dir()
    assert (job_dir / "final").is_dir()
    assert (job_dir / "logs").is_dir()
    assert final_mask.shape == original_ct.shape
    assert final_mask.header.get_data_dtype() == np.dtype(np.uint8)
    assert np.allclose(final_array, np.rint(final_array), atol=0.0, rtol=0.0)


def test_execute_nnunet_failure_includes_stderr_detail(
    client,
    prepared_study,
    create_ai_run,
    tmp_path,
    monkeypatch,
):
    results_dir = write_nnunet_results_tree(
        tmp_path,
        checkpoint_name="checkpoint_best.pth",
    )
    command_path = tmp_path / "venv" / "bin" / "nnUNetv2_predict"

    monkeypatch.setattr(
        "app.ai.nnunet_executor.shutil.which",
        lambda command: str(command_path),
    )

    def fake_run(command, cwd, capture_output, text, env, timeout, check):
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr="RuntimeError: CUDA not available",
        )

    monkeypatch.setattr("app.ai.nnunet_executor.subprocess.run", fake_run)

    client.test_settings.nnunet_enabled = True
    client.test_settings.nnunet_predict_command = str(command_path)
    client.test_settings.nnunet_results_dir = str(results_dir)
    client.test_settings.nnunet_default_dataset = "501"
    client.test_settings.nnunet_default_checkpoint = "checkpoint_best.pth"
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id)

    execute_response = client.post(f"/studies/{study_id}/ai-runs/{run['id']}/execute")
    body = execute_response.json()

    assert execute_response.status_code == 200
    assert body["status"] == "failed"
    assert body["error"].startswith("nnU-Net failed with return code 1:")
    assert body["error_detail"] == "RuntimeError: CUDA not available"


def test_execute_non_nnunet_module_returns_422(client, prepared_study, create_ai_run):
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id, module_id="osteoporosis_hu_analysis")
    response = client.post(f"/studies/{study_id}/ai-runs/{run['id']}/execute")

    assert response.status_code == 422
