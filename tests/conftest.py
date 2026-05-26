import asyncio
import json as jsonlib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import anyio.to_thread
import fastapi.dependencies.utils
import fastapi.routing
import nibabel as nib
import numpy as np
import pytest
import starlette.concurrency
import starlette.datastructures

from app.core.config import Settings, get_settings
from app.main import app


async def direct_run_in_threadpool(func, *args, **kwargs):
    return func(*args, **kwargs)


async def direct_run_sync(func, *args, cancellable=False, limiter=None):
    return func(*args)


fastapi.routing.run_in_threadpool = direct_run_in_threadpool
fastapi.dependencies.utils.run_in_threadpool = direct_run_in_threadpool
starlette.concurrency.run_in_threadpool = direct_run_in_threadpool
starlette.datastructures.run_in_threadpool = direct_run_in_threadpool
anyio.to_thread.run_sync = direct_run_sync


class AsgiResponse:
    def __init__(self, status_code: int, headers: dict[str, str], content: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.content = content
        self.text = content.decode("utf-8", errors="replace")

    def json(self) -> dict:
        return jsonlib.loads(self.content.decode("utf-8"))


class AsgiTestClient:
    def __init__(self, test_app, settings: Settings) -> None:
        self.app = test_app
        self.test_settings = settings

    def get(self, url: str) -> AsgiResponse:
        return self.request("GET", url)

    def post(
        self,
        url: str,
        json: dict | None = None,
        files: dict | None = None,
    ) -> AsgiResponse:
        return self.request("POST", url, json=json, files=files)

    def head(self, url: str) -> AsgiResponse:
        return self.request("HEAD", url)

    def request(
        self,
        method: str,
        url: str,
        json: dict | None = None,
        files: dict | None = None,
    ) -> AsgiResponse:
        body, headers = build_request_body(json=json, files=files)

        return asyncio.run(self._call_app(method, url, body, headers))

    async def _call_app(
        self,
        method: str,
        url: str,
        body: bytes,
        headers: list[tuple[bytes, bytes]],
    ) -> AsgiResponse:
        parsed = urlsplit(url)
        path = unquote(parsed.path)
        messages: list[dict] = []
        sent_request = False
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": parsed.path.encode("ascii"),
            "query_string": parsed.query.encode("ascii"),
            "headers": headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "extensions": {},
        }

        async def receive() -> dict:
            nonlocal sent_request

            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": body, "more_body": False}

            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            messages.append(message)

        await self.app(scope, receive, send)
        return build_response(messages)


def build_request_body(
    json: dict | None = None,
    files: dict | None = None,
) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    if files is not None:
        body, content_type = build_multipart_body(files)
        return body, [
            (b"content-type", content_type.encode("utf-8")),
            (b"content-length", str(len(body)).encode("ascii")),
        ]

    if json is not None:
        body = jsonlib.dumps(json).encode("utf-8")
        return body, [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]

    return b"", [(b"content-length", b"0")]


def build_multipart_body(files: dict) -> tuple[bytes, str]:
    boundary = "test-boundary"
    chunks: list[bytes] = []

    for field_name, value in files.items():
        filename, file_obj, content_type = value
        content = file_obj.read()
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                content,
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def build_response(messages: list[dict]) -> AsgiResponse:
    status_code = 500
    headers: dict[str, str] = {}
    body_chunks: list[bytes] = []

    for message in messages:
        if message["type"] == "http.response.start":
            status_code = message["status"]
            headers = {
                key.decode("latin-1"): value.decode("latin-1")
                for key, value in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            body_chunks.append(message.get("body", b""))

    return AsgiResponse(status_code, headers, b"".join(body_chunks))


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    storage_root = tmp_path / "studies"

    return Settings(
        storage_root=str(storage_root),
        backend_public_url="http://127.0.0.1:8000",
        nnunet_enabled=False,
        nnunet_default_dataset="",
        ct_anatomy_labels_path="",
    )


@pytest.fixture()
def client(test_settings: Settings) -> AsgiTestClient:
    get_settings.cache_clear()
    app.dependency_overrides[get_settings] = lambda: test_settings

    yield AsgiTestClient(app, test_settings)

    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture()
def synthetic_nifti_path(tmp_path: Path) -> Path:
    path = tmp_path / "test1_series.nii.gz"
    array = np.arange(8 * 8 * 8, dtype=np.int16).reshape((8, 8, 8)) - 256
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    image = nib.Nifti1Image(array, affine)

    nib.save(image, str(path))
    return path


@pytest.fixture()
def import_nifti(client: AsgiTestClient, synthetic_nifti_path: Path):
    def _import_nifti() -> dict:
        with synthetic_nifti_path.open("rb") as file:
            response = client.post(
                "/studies/import",
                files={"files": (synthetic_nifti_path.name, file, "application/gzip")},
            )

        assert response.status_code == 200
        return response.json()

    return _import_nifti


@pytest.fixture()
def import_unknown(client: AsgiTestClient, tmp_path: Path):
    def _import_unknown() -> dict:
        path = tmp_path / "unknown.bin"
        path.write_bytes(b"not a medical image")

        with path.open("rb") as file:
            response = client.post(
                "/studies/import",
                files={"files": (path.name, file, "application/octet-stream")},
            )

        assert response.status_code == 200
        return response.json()

    return _import_unknown


@pytest.fixture()
def prepare_study(client: AsgiTestClient):
    def _prepare_study(study_id: str) -> dict:
        response = client.post(f"/studies/{study_id}/prepare")

        assert response.status_code == 200
        return response.json()

    return _prepare_study


@pytest.fixture()
def create_ai_run(client: AsgiTestClient):
    def _create_ai_run(
        study_id: str,
        module_id: str = "ct_anatomy_segmentation_nnunet",
    ) -> dict:
        response = client.post(
            f"/studies/{study_id}/ai-runs",
            json={"module_id": module_id},
        )

        assert response.status_code == 200
        return response.json()

    return _create_ai_run


@pytest.fixture()
def simulate_run(client: AsgiTestClient):
    def _simulate_run(study_id: str, run_id: str) -> dict:
        response = client.post(f"/studies/{study_id}/ai-runs/{run_id}/simulate")

        assert response.status_code == 200
        return response.json()

    return _simulate_run


@pytest.fixture()
def publish_segmentation(client: AsgiTestClient):
    def _publish_segmentation(study_id: str, run_id: str) -> dict:
        response = client.post(
            f"/studies/{study_id}/ai-runs/{run_id}/publish-segmentation"
        )

        assert response.status_code == 200
        return response.json()

    return _publish_segmentation


@pytest.fixture()
def create_hu_analysis(client: AsgiTestClient):
    def _create_hu_analysis(
        study_id: str,
        segmentation_id: str,
        label_ids: list[int] | None = None,
    ) -> dict:
        payload = {
            "module_id": "segmentation_label_hu_statistics",
            "segmentation_id": segmentation_id,
            "roi_mode": "whole_label",
        }

        if label_ids is not None:
            payload["label_ids"] = label_ids

        response = client.post(f"/studies/{study_id}/analyses", json=payload)

        assert response.status_code == 200
        return response.json()

    return _create_hu_analysis


@pytest.fixture()
def prepared_study(import_nifti, prepare_study) -> dict:
    study = import_nifti()
    volume = prepare_study(study["id"])

    return {"study": study, "volume": volume}


@pytest.fixture()
def simulated_run(prepared_study, create_ai_run, simulate_run) -> dict:
    study_id = prepared_study["study"]["id"]
    run = create_ai_run(study_id)
    simulated = simulate_run(study_id, run["id"])

    return {"study_id": study_id, "run": simulated}


@pytest.fixture()
def published_segmentation(simulated_run, publish_segmentation) -> dict:
    study_id = simulated_run["study_id"]
    run = simulated_run["run"]
    segmentation = publish_segmentation(study_id, run["id"])

    return {"study_id": study_id, "run": run, "segmentation": segmentation}
