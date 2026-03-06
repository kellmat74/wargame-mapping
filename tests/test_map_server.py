import pytest

import map_server


def test_read_generator_version_matches_source():
    assert map_server.read_generator_version() == "v3.0.0"


def test_resolve_output_map_dir_accepts_output_subdir(tmp_path):
    project_root = tmp_path / "project"
    valid_dir = project_root / "output" / "country" / "map" / "2026-01-01_12-00_v3.0.0"
    valid_dir.mkdir(parents=True)

    resolved = map_server.resolve_output_map_dir(str(valid_dir), base_path=project_root)
    assert resolved == valid_dir.resolve()


def test_resolve_output_map_dir_rejects_output_root(tmp_path):
    project_root = tmp_path / "project"
    output_root = project_root / "output"
    output_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="inside output/"):
        map_server.resolve_output_map_dir(str(output_root), base_path=project_root)


def test_resolve_output_map_dir_rejects_outside_path(tmp_path):
    project_root = tmp_path / "project"
    (project_root / "output").mkdir(parents=True)
    outside = tmp_path / "not-output"
    outside.mkdir()

    with pytest.raises(ValueError, match="inside output/"):
        map_server.resolve_output_map_dir(str(outside), base_path=project_root)


def test_rerender_map_rejects_invalid_path(client):
    response = client.post("/api/rerender-map", json={"map_path": "/tmp"})
    assert response.status_code == 400
    assert "inside output/" in response.get_json()["error"]


@pytest.fixture
def client():
    map_server.app.config["TESTING"] = True
    return map_server.app.test_client()
