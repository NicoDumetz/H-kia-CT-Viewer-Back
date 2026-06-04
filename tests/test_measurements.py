def test_create_hu_circle_measurement_on_prepared_volume(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.post(
        f"/studies/{study_id}/measurements/hu-circle",
        json={
            "plane": "axial",
            "center_world": [3.0, 3.0, 6.0],
            "radius_mm": 1.1,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["plane"] == "axial"
    assert body["voxel_count"] == 5
    assert body["area_mm2"] == 5.0
    assert body["radius_mm"] == 1.1
    assert body["hu"]["mean"] == -37.0
    assert body["hu"]["median"] == -37.0
    assert body["hu"]["min"] == -101.0
    assert body["hu"]["max"] == 27.0


def test_create_hu_circle_measurement_uses_edge_world(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.post(
        f"/studies/{study_id}/measurements/hu-circle",
        json={
            "plane": "axial",
            "center_world": [3.0, 3.0, 6.0],
            "edge_world": [4.1, 3.0, 6.0],
        },
    )

    assert response.status_code == 200
    assert round(response.json()["radius_mm"], 1) == 1.1


def test_create_hu_circle_measurement_missing_study_returns_404(client):
    response = client.post(
        "/studies/missing/measurements/hu-circle",
        json={
            "plane": "axial",
            "center_world": [0.0, 0.0, 0.0],
            "radius_mm": 1.0,
        },
    )

    assert response.status_code == 404


def test_create_hu_circle_measurement_missing_volume_returns_404(client, import_nifti):
    study = import_nifti()
    response = client.post(
        f"/studies/{study['id']}/measurements/hu-circle",
        json={
            "plane": "axial",
            "center_world": [0.0, 0.0, 0.0],
            "radius_mm": 1.0,
        },
    )

    assert response.status_code == 404


def test_create_hu_circle_measurement_outside_volume_returns_422(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.post(
        f"/studies/{study_id}/measurements/hu-circle",
        json={
            "plane": "axial",
            "center_world": [100.0, 100.0, 100.0],
            "radius_mm": 1.0,
        },
    )

    assert response.status_code == 422
    assert "outside" in response.json()["detail"]


def test_create_hu_circle_measurement_invalid_plane_returns_422(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.post(
        f"/studies/{study_id}/measurements/hu-circle",
        json={
            "plane": "oblique",
            "center_world": [3.0, 3.0, 6.0],
            "radius_mm": 1.0,
        },
    )

    assert response.status_code == 422


def test_create_hu_circle_measurement_invalid_radius_returns_422(client, prepared_study):
    study_id = prepared_study["study"]["id"]
    response = client.post(
        f"/studies/{study_id}/measurements/hu-circle",
        json={
            "plane": "axial",
            "center_world": [3.0, 3.0, 6.0],
            "radius_mm": 0,
        },
    )

    assert response.status_code == 422
