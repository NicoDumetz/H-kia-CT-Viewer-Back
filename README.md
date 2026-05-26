# Hekia CT Viewer Backend

Backend FastAPI minimal pour importer des examens CT au format DICOM, DICOMDIR ou NIfTI.

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
uvicorn app.main:app --reload
```

L'API écoute par défaut sur `http://127.0.0.1:8000`.

## Test health

```bash
curl http://127.0.0.1:8000/health
```

Réponse attendue :

```json
{"status":"ok"}
```

## Tests automatises

```bash
python -m pytest -q
```

## Test import NIfTI

```bash
curl -X POST http://127.0.0.1:8000/studies/import \
  -F "files=@/path/to/scan.nii.gz"
```

## Test import DICOM

```bash
curl -X POST http://127.0.0.1:8000/studies/import \
  -F "files=@/path/to/image-001.dcm"
```

Pour plusieurs fichiers DICOM :

```bash
curl -X POST http://127.0.0.1:8000/studies/import \
  -F "files=@/path/to/image-001.dcm" \
  -F "files=@/path/to/image-002.dcm"
```

Les fichiers importés sont stockés dans `storage/studies/{study_id}/source/`.
