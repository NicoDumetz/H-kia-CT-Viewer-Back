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

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from re import sub

from app.ai.schemas import (
    AiModuleDefinition,
    AiRunArtifactRead,
    AiRunOutputRead,
    AiRunRead,
)
from app.ai.storage import (
    LOGS_FILENAME,
    get_nnunet_input_dir,
    get_nnunet_output_dir,
    get_outputs_dir,
)
from app.core.config import Settings
from app.studies.volume import VOLUME_RELATIVE_PATH


class NnunetExecutionError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


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
    input_dir: Path
    output_dir: Path
    outputs_dir: Path
    expected_output_path: Path
    segmentation_path: Path


def check_nnunet_available(
    module: AiModuleDefinition,
    settings: Settings,
) -> tuple[bool, str | None]:
    config = resolve_nnunet_config(module, settings)
    command_path = shutil.which(config.command)

    if not settings.nnunet_enabled:
        return False, "nnU-Net execution is disabled"

    if not config.dataset:
        return False, "nnU-Net dataset is not configured"

    if command_path is None:
        return False, "nnU-Net predict command is not available in PATH"

    return True, None


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
        device=device,
        timeout_seconds=settings.nnunet_timeout_seconds,
    )


def build_nnunet_case_id(study_id: str) -> str:
    safe_id = sub(r"[^A-Za-z0-9_]", "", study_id)

    return f"case_{safe_id}"


def prepare_nnunet_input(study_dir: Path, run_dir: Path, run: AiRunRead) -> NnunetPreparedPaths:
    case_id = build_nnunet_case_id(run.study_id)
    input_dir = get_nnunet_input_dir(run_dir)
    output_dir = get_nnunet_output_dir(run_dir)
    outputs_dir = get_outputs_dir(run_dir)
    source_volume_path = study_dir / VOLUME_RELATIVE_PATH
    input_volume_path = input_dir / f"{case_id}_0000.nii.gz"
    expected_output_path = output_dir / f"{case_id}.nii.gz"
    segmentation_path = outputs_dir / "segmentation.nii.gz"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_volume_path, input_volume_path)

    return NnunetPreparedPaths(
        case_id=case_id,
        input_dir=input_dir,
        output_dir=output_dir,
        outputs_dir=outputs_dir,
        expected_output_path=expected_output_path,
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
    paths = prepare_nnunet_input(study_dir, run_dir, run)
    command = build_nnunet_command(config, paths)
    start = time.monotonic()

    try:
        result = subprocess.run(
            command,
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration_seconds = time.monotonic() - start
        write_nnunet_logs(run_dir, command, None, exc.stdout, exc.stderr, duration_seconds)
        raise NnunetExecutionError("nnU-Net execution timed out") from exc
    except OSError as exc:
        duration_seconds = time.monotonic() - start
        write_nnunet_logs(run_dir, command, None, "", str(exc), duration_seconds)
        raise NnunetExecutionError(f"Failed to start nnU-Net: {exc}") from exc

    duration_seconds = time.monotonic() - start
    write_nnunet_logs(
        run_dir,
        command,
        result.returncode,
        result.stdout,
        result.stderr,
        duration_seconds,
    )

    if result.returncode != 0:
        raise NnunetExecutionError(f"nnU-Net failed with return code {result.returncode}")

    return collect_nnunet_output(run, paths)


def build_nnunet_command(
    config: ResolvedNnunetConfig,
    paths: NnunetPreparedPaths,
) -> list[str]:
    return [
        config.command,
        "-i",
        str(paths.input_dir),
        "-o",
        str(paths.output_dir),
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
    ]


def collect_nnunet_output(run: AiRunRead, paths: NnunetPreparedPaths) -> AiRunOutputRead:
    result_relative_path = f"derived/ai-runs/{run.id}/outputs/segmentation.nii.gz"
    logs_relative_path = f"derived/ai-runs/{run.id}/{LOGS_FILENAME}"

    if not paths.expected_output_path.is_file():
        raise NnunetExecutionError("nnU-Net output segmentation was not found")

    shutil.copy2(paths.expected_output_path, paths.segmentation_path)

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
    command: list[str],
    returncode: int | None,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    duration_seconds: float,
) -> Path:
    logs_path = run_dir / LOGS_FILENAME
    stdout_text = decode_output(stdout)
    stderr_text = decode_output(stderr)
    content = (
        f"command={' '.join(command)}\n"
        f"cwd={run_dir}\n"
        f"returncode={returncode}\n"
        f"duration_seconds={duration_seconds:.3f}\n"
        "\nstdout:\n"
        f"{stdout_text}\n"
        "\nstderr:\n"
        f"{stderr_text}\n"
    )

    logs_path.write_text(content, encoding="utf-8")
    return logs_path


def decode_output(output: str | bytes | None) -> str:
    if output is None:
        return ""

    if isinstance(output, bytes):
        return output.decode(errors="replace")

    return output
