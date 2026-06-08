# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : nnunet_executor.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

import os
import shutil
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from re import sub

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from app.ai.schemas import (
    AiModuleDefinition,
    AiRunArtifactRead,
    AiRunOutputRead,
    AiRunRead,
)
from app.ai.storage import (
    LOGS_FILENAME,
    get_outputs_dir,
)
from app.core.config import Settings
from app.studies.volume import VOLUME_RELATIVE_PATH


class NnunetExecutionError(Exception):
    def __init__(self, message: str, error_detail: str | None = None) -> None:
        self.message = message
        self.error_detail = error_detail
        super().__init__(message)


MIN_CHECKPOINT_SIZE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class ResolvedNnunetConfig:
    command: str
    dataset: str
    configuration: str
    fold: str
    checkpoint: str
    device: str
    timeout_seconds: int


@dataclass(frozen=True)
class NnunetPreparedPaths:
    case_id: str
    job_dir: Path
    input_dir: Path
    lowres_dir: Path
    output_lowres_dir: Path
    final_dir: Path
    logs_dir: Path
    outputs_dir: Path
    original_volume_path: Path
    input_volume_path: Path
    lowres_volume_path: Path
    expected_output_path: Path
    final_job_segmentation_path: Path
    segmentation_path: Path


@dataclass(frozen=True)
class NnunetModelArtifacts:
    results_dir: Path
    dataset_dir: Path
    configuration_dir: Path
    fold_dir: Path
    plans_path: Path
    dataset_json_path: Path
    checkpoint_path: Path


def check_nnunet_available(
    module: AiModuleDefinition,
    settings: Settings,
) -> tuple[bool, str | None]:
    config = resolve_nnunet_config(module, settings)

    if not settings.nnunet_enabled:
        return False, "NNUNET_ENABLED=false"

    artifacts_available, artifacts_reason = check_nnunet_model_artifacts(config, settings)

    if not artifacts_available:
        return False, artifacts_reason

    if not config.command:
        return False, "NNUNET_PREDICT_COMMAND is not configured"

    try:
        resolve_nnunet_command_path(config.command)
    except NnunetExecutionError as exc:
        return False, exc.message

    return True, None


def check_nnunet_model_artifacts(
    config: ResolvedNnunetConfig,
    settings: Settings,
) -> tuple[bool, str | None]:
    try:
        resolve_nnunet_model_artifacts(config, settings)
    except NnunetExecutionError as exc:
        return False, exc.message

    return True, None


def resolve_nnunet_model_artifacts(
    config: ResolvedNnunetConfig,
    settings: Settings,
) -> NnunetModelArtifacts:
    results_dir = settings.nnunet_results_dir

    if not results_dir:
        raise NnunetExecutionError("NNUNET_RESULTS_DIR is not configured")

    if not config.dataset:
        raise NnunetExecutionError("NNUNET_DEFAULT_DATASET is not configured")

    if not config.configuration:
        raise NnunetExecutionError("NNUNET_DEFAULT_CONFIGURATION is not configured")

    if not config.fold:
        raise NnunetExecutionError("NNUNET_DEFAULT_FOLD is not configured")

    if not config.checkpoint:
        raise NnunetExecutionError("NNUNET_DEFAULT_CHECKPOINT is not configured")

    results_path = resolve_absolute_path(Path(results_dir))

    if not results_path.is_dir():
        raise NnunetExecutionError(f"NNUNET_RESULTS_DIR does not exist: {results_path}")

    dataset_dir = resolve_nnunet_dataset_dir(results_path, config.dataset)
    configuration_dir = resolve_nnunet_configuration_dir(dataset_dir, config.configuration)
    plans_path = configuration_dir / "plans.json"
    dataset_json_path = configuration_dir / "dataset.json"
    fold_dir = configuration_dir / f"fold_{config.fold}"
    checkpoint_path = fold_dir / config.checkpoint

    if not plans_path.is_file():
        raise NnunetExecutionError(f"Missing plans.json: {plans_path}")

    if not dataset_json_path.is_file():
        raise NnunetExecutionError(f"Missing dataset.json: {dataset_json_path}")

    if not fold_dir.is_dir():
        raise NnunetExecutionError(f"Fold directory not found: {fold_dir}")

    if not checkpoint_path.is_file():
        raise NnunetExecutionError(f"Missing checkpoint: {checkpoint_path}")

    validate_nnunet_checkpoint(checkpoint_path)

    return NnunetModelArtifacts(
        results_dir=results_path,
        dataset_dir=dataset_dir,
        configuration_dir=configuration_dir,
        fold_dir=fold_dir,
        plans_path=plans_path,
        dataset_json_path=dataset_json_path,
        checkpoint_path=checkpoint_path,
    )


def resolve_nnunet_dataset_dir(results_dir: Path, dataset: str) -> Path:
    exact_dir = results_dir / dataset

    if exact_dir.is_dir():
        return exact_dir

    if dataset.isdigit():
        candidates = sorted(
            path
            for path in results_dir.glob(f"Dataset{dataset}_*")
            if path.is_dir()
        )

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise NnunetExecutionError(
                f"Multiple nnU-Net dataset directories found for dataset '{dataset}': {names}"
            )

        expected = results_dir / f"Dataset{dataset}_*"
        raise NnunetExecutionError(
            f"Dataset directory not found for dataset '{dataset}' under {results_dir}; "
            f"expected {expected}"
        )

    raise NnunetExecutionError(f"Dataset directory not found: {exact_dir}")


def resolve_nnunet_configuration_dir(dataset_dir: Path, configuration: str) -> Path:
    expected_dir = dataset_dir / f"nnUNetTrainer__nnUNetPlans__{configuration}"

    if expected_dir.is_dir():
        return expected_dir

    candidates = sorted(
        path
        for path in dataset_dir.glob(f"*__{configuration}")
        if path.is_dir()
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise NnunetExecutionError(
            f"Multiple nnU-Net configuration directories found for configuration "
            f"'{configuration}': {names}"
        )

    raise NnunetExecutionError(f"Configuration directory not found: {expected_dir}")


def resolve_nnunet_config(
    module: AiModuleDefinition,
    settings: Settings,
) -> ResolvedNnunetConfig:
    module_config = module.nnunet
    dataset = module_config.dataset if module_config and module_config.dataset else settings.nnunet_default_dataset
    configuration = (
        module_config.configuration
        if module_config and module_config.configuration
        else settings.nnunet_default_configuration
    )
    fold = module_config.fold if module_config and module_config.fold else settings.nnunet_default_fold
    checkpoint = (
        module_config.checkpoint
        if module_config and module_config.checkpoint
        else settings.nnunet_default_checkpoint
    )
    device = module_config.device if module_config and module_config.device else settings.nnunet_default_device

    return ResolvedNnunetConfig(
        command=settings.nnunet_predict_command,
        dataset=dataset,
        configuration=configuration,
        fold=fold,
        checkpoint=checkpoint,
        device=device.strip().lower(),
        timeout_seconds=settings.nnunet_timeout_seconds,
    )


def build_nnunet_case_id(study_id: str) -> str:
    safe_id = sub(r"[^A-Za-z0-9_]", "", study_id)

    return f"case_{safe_id}"


def prepare_nnunet_input(
    study_dir: Path,
    run_dir: Path,
    run: AiRunRead,
    settings: Settings,
) -> NnunetPreparedPaths:
    case_id = build_nnunet_case_id(run.study_id)
    study_dir = resolve_absolute_path(study_dir)
    run_dir = resolve_absolute_path(run_dir)
    job_dir = study_dir / "tmp" / "jobs" / run.id
    input_dir = job_dir / "input"
    lowres_dir = job_dir / "lowres"
    output_lowres_dir = job_dir / "output_lowres"
    final_dir = job_dir / "final"
    logs_dir = job_dir / "logs"
    outputs_dir = get_outputs_dir(run_dir)
    source_volume_path = study_dir / VOLUME_RELATIVE_PATH
    input_volume_path = input_dir / f"{case_id}_0000.nii.gz"
    lowres_volume_path = lowres_dir / f"{case_id}_0000.nii.gz"
    expected_output_path = output_lowres_dir / f"{case_id}.nii.gz"
    final_job_segmentation_path = final_dir / "mask.nii.gz"
    segmentation_path = outputs_dir / "segmentation.nii.gz"

    shutil.rmtree(job_dir, ignore_errors=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    lowres_dir.mkdir(parents=True, exist_ok=True)
    output_lowres_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    write_original_volume_reference(source_volume_path, input_volume_path)
    write_lowres_volume(source_volume_path, lowres_volume_path, settings.nnunet_lowres_spacing_mm)
    validate_lowres_input(paths_lowres_dir=lowres_dir, expected_input_path=lowres_volume_path)

    return NnunetPreparedPaths(
        case_id=case_id,
        job_dir=job_dir,
        input_dir=input_dir,
        lowres_dir=lowres_dir,
        output_lowres_dir=output_lowres_dir,
        final_dir=final_dir,
        logs_dir=logs_dir,
        outputs_dir=outputs_dir,
        original_volume_path=source_volume_path,
        input_volume_path=input_volume_path,
        lowres_volume_path=lowres_volume_path,
        expected_output_path=expected_output_path,
        final_job_segmentation_path=final_job_segmentation_path,
        segmentation_path=segmentation_path,
    )


def run_nnunet_prediction(
    run: AiRunRead,
    study_dir: Path,
    run_dir: Path,
    module: AiModuleDefinition,
    settings: Settings,
) -> AiRunOutputRead:
    config = resolve_nnunet_config(module, settings)
    model_artifacts = resolve_nnunet_model_artifacts(config, settings)
    cwd = get_backend_root()
    study_dir = resolve_absolute_path(study_dir)
    run_dir = resolve_absolute_path(run_dir)
    paths = prepare_nnunet_input(study_dir, run_dir, run, settings)
    command = build_nnunet_command(config, paths)
    env = build_nnunet_env(settings)
    validate_lowres_input(
        paths_lowres_dir=paths.lowres_dir,
        expected_input_path=paths.lowres_volume_path,
    )
    start = time.monotonic()

    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            env=env,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration_seconds = time.monotonic() - start
        write_nnunet_logs(
            run_dir,
            paths,
            command,
            None,
            exc.stdout,
            exc.stderr,
            duration_seconds,
            cwd=cwd,
            env=env,
            config=config,
            model_artifacts=model_artifacts,
        )
        error_detail = extract_error_detail(exc.stderr, exc.stdout)
        message = append_error_detail("nnU-Net execution timed out", error_detail)
        raise NnunetExecutionError(message, error_detail=error_detail) from exc
    except OSError as exc:
        duration_seconds = time.monotonic() - start
        write_nnunet_logs(
            run_dir,
            paths,
            command,
            None,
            "",
            str(exc),
            duration_seconds,
            cwd=cwd,
            env=env,
            config=config,
            model_artifacts=model_artifacts,
        )
        error_detail = str(exc)
        raise NnunetExecutionError(
            f"Failed to start nnU-Net: {error_detail}",
            error_detail=error_detail,
        ) from exc

    duration_seconds = time.monotonic() - start
    write_nnunet_logs(
        run_dir,
        paths,
        command,
        result.returncode,
        result.stdout,
        result.stderr,
        duration_seconds,
        cwd=cwd,
        env=env,
        config=config,
        model_artifacts=model_artifacts,
    )

    if result.returncode != 0:
        error_detail = extract_error_detail(result.stderr, result.stdout)
        message = append_error_detail(
            f"nnU-Net failed with return code {result.returncode}",
            error_detail,
        )
        raise NnunetExecutionError(message, error_detail=error_detail)

    return collect_nnunet_output(run, paths)


def build_nnunet_command(
    config: ResolvedNnunetConfig,
    paths: NnunetPreparedPaths,
) -> list[str]:
    command_path = resolve_nnunet_command_path(config.command)

    return [
        str(command_path),
        "-i",
        str(paths.lowres_dir),
        "-o",
        str(paths.output_lowres_dir),
        "-d",
        config.dataset,
        "-c",
        config.configuration,
        "-f",
        config.fold,
        "-chk",
        config.checkpoint,
        "-device",
        config.device,
        "--disable_tta",
        "-npp",
        "1",
        "-nps",
        "1",
    ]


def build_nnunet_env(settings: Settings) -> dict[str, str]:
    env = dict(os.environ)
    backend_root = get_backend_root()
    raw_dir = backend_root / "storage" / "nnunet" / "raw"
    preprocessed_dir = backend_root / "storage" / "nnunet" / "preprocessed"

    if not settings.nnunet_results_dir:
        raise NnunetExecutionError("NNUNET_RESULTS_DIR is not configured")

    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    env["nnUNet_raw"] = str(raw_dir)
    env["nnUNet_preprocessed"] = str(preprocessed_dir)
    env["nnUNet_results"] = str(resolve_absolute_path(Path(settings.nnunet_results_dir)))
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["PYTHONNOUSERSITE"] = "1"

    return env


def collect_nnunet_output(run: AiRunRead, paths: NnunetPreparedPaths) -> AiRunOutputRead:
    result_relative_path = f"derived/ai-runs/{run.id}/outputs/segmentation.nii.gz"
    logs_relative_path = f"derived/ai-runs/{run.id}/{LOGS_FILENAME}"

    if not paths.expected_output_path.is_file():
        raise NnunetExecutionError(
            f"Output mask not found after prediction: {paths.expected_output_path}"
        )

    write_final_mask_on_original_grid(
        lowres_mask_path=paths.expected_output_path,
        original_volume_path=paths.original_volume_path,
        final_job_path=paths.final_job_segmentation_path,
        output_path=paths.segmentation_path,
    )
    validate_final_segmentation(paths.segmentation_path, paths.original_volume_path)

    return AiRunOutputRead(
        result_path=result_relative_path,
        artifacts=[
            AiRunArtifactRead(
                type="nifti_segmentation",
                name="segmentation.nii.gz",
                relative_path=result_relative_path,
                url=f"/studies/{run.study_id}/files/{result_relative_path}",
            ),
            AiRunArtifactRead(
                type="text",
                name=LOGS_FILENAME,
                relative_path=logs_relative_path,
                url=f"/studies/{run.study_id}/files/{logs_relative_path}",
            ),
        ],
    )


def write_nnunet_logs(
    run_dir: Path,
    paths: NnunetPreparedPaths,
    command: list[str],
    returncode: int | None,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    duration_seconds: float,
    cwd: Path,
    env: dict[str, str],
    config: ResolvedNnunetConfig,
    model_artifacts: NnunetModelArtifacts | None,
) -> Path:
    logs_path = run_dir / LOGS_FILENAME
    stdout_text = decode_output(stdout)
    stderr_text = decode_output(stderr)
    checkpoint_path = model_artifacts.checkpoint_path if model_artifacts else ""
    checkpoint_size = (
        model_artifacts.checkpoint_path.stat().st_size
        if model_artifacts and model_artifacts.checkpoint_path.exists()
        else ""
    )
    content = (
        f"command={shlex.join(command)}\n"
        f"cwd={cwd}\n"
        f"job_dir={paths.job_dir}\n"
        f"input_dir={paths.input_dir}\n"
        f"lowres_dir={paths.lowres_dir}\n"
        f"output_lowres_dir={paths.output_lowres_dir}\n"
        f"final_dir={paths.final_dir}\n"
        f"logs_dir={paths.logs_dir}\n"
        f"input_volume={paths.input_volume_path}\n"
        f"lowres_input={paths.lowres_volume_path}\n"
        f"nnUNet_raw={env.get('nnUNet_raw', '')}\n"
        f"nnUNet_preprocessed={env.get('nnUNet_preprocessed', '')}\n"
        f"nnUNet_results={env.get('nnUNet_results', '')}\n"
        f"device={config.device}\n"
        f"checkpoint={config.checkpoint}\n"
        f"checkpoint_path={checkpoint_path}\n"
        f"checkpoint_size_bytes={checkpoint_size}\n"
        f"returncode={returncode}\n"
        f"duration_seconds={duration_seconds:.3f}\n"
        "\nstdout:\n"
        f"{stdout_text}\n"
        "\nstderr:\n"
        f"{stderr_text}\n"
    )

    logs_path.write_text(content, encoding="utf-8")
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    (paths.logs_dir / LOGS_FILENAME).write_text(content, encoding="utf-8")
    return logs_path


def resolve_absolute_path(path: Path) -> Path:
    return path.expanduser().resolve()


def get_backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_nnunet_command_path(command: str) -> Path:
    command_path = shutil.which(command)

    if command_path is None:
        if Path(command).is_absolute():
            raise NnunetExecutionError(
                f"nnU-Net predict command not found or not executable: {command}"
            )

        raise NnunetExecutionError(f"nnU-Net predict command not found in PATH: {command}")

    resolved_path = Path(command_path).expanduser().resolve()

    if is_user_local_nnunet_command(resolved_path):
        raise NnunetExecutionError(
            f"Refusing to use user-local nnU-Net command: {resolved_path}. "
            "Configure NNUNET_PREDICT_COMMAND to the backend .venv-nnunet binary."
        )

    return resolved_path


def is_user_local_nnunet_command(command_path: Path) -> bool:
    return (
        command_path.name == "nnUNetv2_predict"
        and len(command_path.parts) >= 3
        and command_path.parts[-3:] == (".local", "bin", "nnUNetv2_predict")
    )


def validate_nnunet_checkpoint(checkpoint_path: Path) -> None:
    try:
        size = checkpoint_path.stat().st_size

        if size < MIN_CHECKPOINT_SIZE_BYTES:
            raise OSError(f"checkpoint is too small ({size} bytes)")

        with checkpoint_path.open("rb") as checkpoint_file:
            if checkpoint_file.read(1) == b"":
                raise OSError("checkpoint is empty")

            checkpoint_file.seek(size - 1)

            if checkpoint_file.read(1) == b"":
                raise OSError("checkpoint cannot be read to the end")
    except OSError as exc:
        raise NnunetExecutionError(
            f"Checkpoint file appears corrupted or unreadable: {checkpoint_path}"
        ) from exc


def validate_lowres_input(paths_lowres_dir: Path, expected_input_path: Path) -> None:
    if not paths_lowres_dir.is_dir():
        raise NnunetExecutionError(f"nnU-Net lowres input directory not found: {paths_lowres_dir}")

    lowres_inputs = sorted(paths_lowres_dir.glob("*_0000.nii.gz"))

    if not lowres_inputs:
        raise NnunetExecutionError(
            f"nnU-Net lowres input directory contains no *_0000.nii.gz file: "
            f"{paths_lowres_dir}"
        )

    if expected_input_path not in lowres_inputs:
        raise NnunetExecutionError(
            f"Expected nnU-Net lowres input was not found: {expected_input_path}"
        )


def write_original_volume_reference(source_volume_path: Path, input_volume_path: Path) -> None:
    if not source_volume_path.is_file():
        raise NnunetExecutionError(f"Input CT volume not found: {source_volume_path}")

    try:
        if input_volume_path.exists() or input_volume_path.is_symlink():
            input_volume_path.unlink()

        os.symlink(source_volume_path, input_volume_path)
    except OSError:
        try:
            shutil.copy2(source_volume_path, input_volume_path)
        except OSError as exc:
            raise NnunetExecutionError(
                f"Failed to stage original CT volume for nnU-Net: {exc}"
            ) from exc


def write_lowres_volume(
    source_volume_path: Path,
    lowres_volume_path: Path,
    spacing_mm: float,
) -> None:
    try:
        image = sitk.ReadImage(str(source_volume_path))
        resampled = resample_ct_image(image, (spacing_mm, spacing_mm, spacing_mm))
        sitk.WriteImage(resampled, str(lowres_volume_path))
    except Exception as exc:
        raise NnunetExecutionError(f"Failed to resample CT volume to lowres: {exc}") from exc


def resample_ct_image(image: sitk.Image, spacing: tuple[float, float, float]) -> sitk.Image:
    original_size = np.asarray(image.GetSize(), dtype=float)
    original_spacing = np.asarray(image.GetSpacing(), dtype=float)
    new_spacing = np.asarray(spacing, dtype=float)
    new_size = np.maximum(np.round(original_size * original_spacing / new_spacing), 1).astype(int)
    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(tuple(float(value) for value in new_spacing))
    resampler.SetSize([int(value) for value in new_size])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))

    return resampler.Execute(image)


def write_final_mask_on_original_grid(
    lowres_mask_path: Path,
    original_volume_path: Path,
    final_job_path: Path,
    output_path: Path,
) -> None:
    try:
        lowres_mask = sitk.ReadImage(str(lowres_mask_path))
        original = sitk.ReadImage(str(original_volume_path))
        resampler = sitk.ResampleImageFilter()

        resampler.SetReferenceImage(original)
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        final_mask = resampler.Execute(lowres_mask)
        sitk.WriteImage(final_mask, str(final_job_path))
        save_sitk_mask_with_original_affine(final_mask, original_volume_path, output_path)
    except Exception as exc:
        raise NnunetExecutionError(
            f"Failed to resample nnU-Net mask to original CT grid: {exc}"
        ) from exc


def save_sitk_mask_with_original_affine(
    image: sitk.Image,
    original_volume_path: Path,
    output_path: Path,
) -> None:
    original_nib = nib.load(str(original_volume_path))
    array_zyx = sitk.GetArrayFromImage(image)
    array_xyz = np.transpose(array_zyx, (2, 1, 0))

    if not is_integer_label_array(array_xyz):
        raise NnunetExecutionError("nnU-Net mask contains non-integer labels after resampling")

    label_array = np.rint(array_xyz).astype(np.int16)
    header = original_nib.header.copy()
    header.set_data_dtype(label_array.dtype)
    nib.save(nib.Nifti1Image(label_array, original_nib.affine, header), str(output_path))


def validate_final_segmentation(segmentation_path: Path, original_volume_path: Path) -> None:
    segmentation = nib.load(str(segmentation_path))
    original = nib.load(str(original_volume_path))
    array = np.asanyarray(segmentation.dataobj)

    if segmentation.shape != original.shape:
        raise NnunetExecutionError(
            f"Final nnU-Net mask shape {list(segmentation.shape)} does not match "
            f"CT shape {list(original.shape)}"
        )

    if not is_integer_label_array(array):
        raise NnunetExecutionError("Final nnU-Net mask contains non-integer labels")

    if not np.allclose(segmentation.affine, original.affine, atol=1e-3, rtol=1e-3):
        raise NnunetExecutionError("Final nnU-Net mask affine is not compatible with CT")


def is_integer_label_array(array: np.ndarray) -> bool:
    if np.issubdtype(array.dtype, np.integer):
        return True

    if not bool(np.all(np.isfinite(array))):
        return False

    return bool(np.allclose(array, np.rint(array), atol=0.0, rtol=0.0))


def decode_output(output: str | bytes | None) -> str:
    if output is None:
        return ""

    if isinstance(output, bytes):
        return output.decode(errors="replace")

    return output


def extract_error_detail(stderr: str | bytes | None, stdout: str | bytes | None) -> str | None:
    text = decode_output(stderr).strip() or decode_output(stdout).strip()

    if not text:
        return None

    lines = [line for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-40:]).strip()

    if len(tail) > 4000:
        tail = tail[-4000:].lstrip()

    return tail or None


def append_error_detail(message: str, error_detail: str | None) -> str:
    if not error_detail:
        return message

    return f"{message}: {error_detail}"
