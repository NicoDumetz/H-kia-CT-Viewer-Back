# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : registry.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

import shutil

from app.ai.schemas import AiModuleDefinition, AiModuleListResponse, AiModuleNnunetConfig, AiModuleRead
from app.core.config import Settings


AI_MODULES = {
    "ct_anatomy_segmentation_nnunet": AiModuleDefinition(
        id="ct_anatomy_segmentation_nnunet",
        name="CT anatomy segmentation nnU-Net",
        task_type="segmentation",
        description="Segment multiple anatomical structures from a prepared CT NIfTI volume.",
        input_type="prepared_volume",
        output_type="multi_label_segmentation_mask",
        is_available=False,
        runner="nnunet",
        labels=None,
        nnunet=AiModuleNnunetConfig(
            dataset=None,
            configuration=None,
            fold=None,
            checkpoint=None,
            device=None,
        ),
    ),
    "osteoporosis_hu_analysis": AiModuleDefinition(
        id="osteoporosis_hu_analysis",
        name="Osteoporosis HU analysis",
        task_type="measurement",
        description="Compute HU metrics on selected vertebral ROI from a CT segmentation.",
        input_type="multi_label_segmentation_mask",
        output_type="hu_metrics",
        is_available=False,
        runner="internal",
        labels=None,
        nnunet=None,
    ),
    "segmentation_label_hu_statistics": AiModuleDefinition(
        id="segmentation_label_hu_statistics",
        name="Segmentation label HU statistics",
        task_type="measurement",
        description="Compute HU and volume statistics for labels from a published CT segmentation.",
        input_type="prepared_volume_and_segmentation",
        output_type="label_hu_statistics",
        is_available=True,
        runner="internal",
        labels=None,
        nnunet=None,
    ),
}


def list_ai_module_definitions() -> list[AiModuleDefinition]:
    return list(AI_MODULES.values())


def list_modules(settings: Settings) -> AiModuleListResponse:
    items = [to_module_read(module, settings) for module in list_ai_module_definitions()]

    return AiModuleListResponse(items=items)


def get_module(module_id: str) -> AiModuleDefinition | None:
    return AI_MODULES.get(module_id)


def to_module_read(module: AiModuleDefinition, settings: Settings) -> AiModuleRead:
    return AiModuleRead(
        id=module.id,
        name=module.name,
        task_type=module.task_type,
        description=module.description,
        input_type=module.input_type,
        output_type=module.output_type,
        is_available=is_module_available(module, settings),
        runner=module.runner,
        labels=module.labels,
    )


def is_module_available(module: AiModuleDefinition, settings: Settings) -> bool:
    dataset = resolve_module_dataset(module, settings)
    command_path = shutil.which(settings.nnunet_predict_command)

    if module.id == "segmentation_label_hu_statistics":
        return True

    if module.runner != "nnunet":
        return False

    return settings.nnunet_enabled and command_path is not None and bool(dataset)


def resolve_module_dataset(module: AiModuleDefinition, settings: Settings) -> str:
    if module.nnunet and module.nnunet.dataset:
        return module.nnunet.dataset

    return settings.nnunet_default_dataset
