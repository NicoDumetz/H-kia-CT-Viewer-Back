# =============================================================
#
# ██╗  ██╗███████╗██╗  ██╗██╗ █████╗
# ██║  ██║██╔════╝██║ ██╔╝██║██╔══██╗
# ███████║█████╗  █████╔╝ ██║███████║
# ██╔══██║██╔══╝  ██╔═██╗ ██║██╔══██║
# ██║  ██║███████╗██║  ██╗██║██║  ██║
# ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#
# File        : config.py
# Project     : H-kia-CT-Viewer-Back
# Author      : Nicolas Dumetz
#
# Created     : Tuesday May 26 2026
#
# =============================================================

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    storage_root: str = "storage/studies"
    backend_public_url: str = "http://127.0.0.1:8000"
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    cors_allow_origin_regex: str = (
        r"^https?://("
        r"localhost|127\.0\.0\.1|0\.0\.0\.0|"
        r"10(?:\.\d{1,3}){3}|"
        r"192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}"
        r")(?::\d+)?$"
    )
    nnunet_enabled: bool = False
    nnunet_predict_command: str = "nnUNetv2_predict"
    nnunet_results_dir: str = ""
    nnunet_model_dir: str = "model"
    nnunet_default_dataset: str = ""
    nnunet_default_configuration: str = "3d_lowres"
    nnunet_default_fold: str = "0"
    nnunet_default_checkpoint: str = "checkpoint_best.pth"
    nnunet_default_device: str = "cpu"
    nnunet_lowres_spacing_mm: float = 1.9001551220814243
    nnunet_timeout_seconds: int = 3600
    nnunet_num_preprocessing_processes: int = 1
    nnunet_num_segmentation_export_processes: int = 1
    ct_anatomy_labels_path: str = ""
    storage_tmp_ttl_hours: int = 24

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_cors_allow_origins(settings: Settings) -> list[str]:
    return [
        origin.strip()
        for origin in settings.cors_allow_origins.split(",")
        if origin.strip()
    ]
