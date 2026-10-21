"""API tests for model-squeezer. Uses a throwaway SQ_DATA dir per module."""
import importlib
import os
import sys
import time

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)


@pytest.fixture(scope="module")
def c(tmp_path_factory):
    d = tmp_path_factory.mktemp("sqdata")
    os.environ["SQ_DATA"] = str(d)
    sys.path.insert(0, os.path.join(REPO, "app"))
    import main as app_main
    importlib.reload(app_main)
    from fastapi.testclient import TestClient
    with TestClient(app_main.app) as c:
        yield c, app_main


def wait_job(c, jid, timeout=10):
    """Poll until the background thread settles the job."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = c.get("/api/jobs/%s" % jid).json()
        if j.get("ok") is not None:
            return j
        time.sleep(0.2)
    raise AssertionError("job %s did not settle in %ss" % (jid, timeout))


def test_health(c):
    client, _ = c
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_levels(c):
    client, _ = c
    r = client.get("/api/levels")
    assert r.status_code == 200
    assert len(r.json()["levels"]) == 7


def test_stats_empty(c):
    client, _ = c
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["models"] == 0
    assert body["history_entries"] == 0
    assert "quantize_available" in body


def test_chromebook_plan_needs_params(c):
    client, _ = c
    r = client.get("/api/chromebook/plan")
    assert r.status_code == 400


def test_chromebook_plan_bad_model(c):
    client, _ = c
    r = client.get("/api/chromebook/plan?model_id=nope")
    assert r.status_code == 404


def test_chromebook_plan_params(c):
    client, _ = c
    r = client.get("/api/chromebook/plan?params_b=7")
    assert r.status_code == 200
    body = r.json()
    assert len(body["plan"]) == 7
    assert "4GB" in body["verdict"]


def test_estimates_404(c):
    client, _ = c
    assert client.get("/api/models/nope/estimates").status_code == 404


def test_delete_model_404(c):
    client, _ = c
    assert client.delete("/api/models/nope").status_code == 404


def test_history_clear(c):
    client, _ = c
    r = client.delete("/api/history")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert client.get("/api/history").json()["history"] == []


def test_jobs_list(c):
    client, _ = c
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert isinstance(r.json()["jobs"], list)


def test_compress_unknown_model_fails_job(c):
    client, _ = c
    r = client.post("/api/compress", json={"model_id": "nope", "level": "Q4"})
    assert r.status_code == 200
    job = wait_job(client, r.json()["job_id"])
    assert job["ok"] is False
    assert "model not found" in job["error"]


def test_compress_level_normalized_before_lookup(c):
    client, _ = c
    # "q4_k_m" normalizes to Q4; the job then fails on the model, not the level
    r = client.post("/api/compress", json={"model_id": "nope", "level": "q4_k_m"})
    job = wait_job(client, r.json()["job_id"])
    assert job["ok"] is False
    assert "model not found" in job["error"]


def test_compress_bogus_level_fails_job(c):
    client, _ = c
    r = client.post("/api/compress", json={"model_id": "nope", "level": "Q99"})
    job = wait_job(client, r.json()["job_id"])
    assert job["ok"] is False
    assert "unknown level" in job["error"]


def _upload(client, name, data):
    r = client.post("/api/upload", files={"file": (name, data)})
    assert r.status_code == 200
    job = wait_job(client, r.json()["job_id"])
    assert job["ok"] is True, job
    return job["result"]["model"]


def test_upload_delete_roundtrip(c):
    client, app_main = c
    model = _upload(client, "tiny.gguf", b"GGUF fake bytes")
    mid = model["id"]

    r = client.get("/api/models")
    ids = [m["id"] for m in r.json()["models"]]
    assert mid in ids

    r = client.get("/api/models/%s/estimates" % mid)
    assert r.status_code == 200
    # fake gguf has no readable params -> honest note, no estimates
    assert r.json()["estimates"] == []
    assert r.json()["note"] is not None

    r = client.delete("/api/models/%s" % mid)
    assert r.status_code == 200
    assert r.json()["deleted"] == mid
    assert client.delete("/api/models/%s" % mid).status_code == 404


def test_delete_model_with_children_409(c):
    client, app_main = c
    parent = _upload(client, "parent.gguf", b"GGUF parent")
    reg = app_main.registry()
    reg["child1"] = {"id": "child1", "name": "child.gguf", "path": "/tmp/x",
                     "size_mb": 1, "source": "compressed",
                     "created": "2026-01-01T00:00:00", "origin_id": parent["id"]}
    app_main.save_registry(reg)

    r = client.delete("/api/models/%s" % parent["id"])
    assert r.status_code == 409
    assert "delete those first" in r.json()["detail"]

    assert client.delete("/api/models/child1").status_code == 200
    assert client.delete("/api/models/%s" % parent["id"]).status_code == 200


def test_train_validation(c):
    client, _ = c
    model = _upload(client, "base.bin", b"fake")
    base = {"model_id": model["id"], "method": "lora", "epochs": "1",
            "lr": "0.0002", "batch_size": "1", "grad_accum": "8"}
    files = {"data": ("d.txt", b"hello world")}

    bad = dict(base, epochs="0")
    assert client.post("/api/train", data=bad, files=files).status_code == 422
    bad = dict(base, epochs="99")
    assert client.post("/api/train", data=bad, files=files).status_code == 422
    bad = dict(base, lr="0")
    assert client.post("/api/train", data=bad, files=files).status_code == 422
    bad = dict(base, method="sft")
    assert client.post("/api/train", data=bad, files=files).status_code == 422
    bad = dict(base, batch_size="99")
    assert client.post("/api/train", data=bad, files=files).status_code == 422

    r = client.post("/api/train", data=dict(base, model_id="nope"), files=files)
    assert r.status_code == 404

    empty_files = {"data": ("empty.txt", b"")}
    r = client.post("/api/train", data=base, files=empty_files)
    assert r.status_code == 422


def test_fetch_bad_kind(c):
    client, _ = c
    r = client.post("/api/fetch", json={"kind": "ftp", "url": "x"})
    assert r.status_code == 400


def test_fetch_needs_filename(c):
    client, _ = c
    r = client.post("/api/fetch", json={"kind": "github", "url": "x", "filename": ""})
    assert r.status_code == 400


def test_fetch_list_bad_kind(c):
    client, _ = c
    assert client.get("/api/fetch/list?kind=nope&url=x").status_code == 400


def test_job_404(c):
    client, _ = c
    assert client.get("/api/jobs/nope").status_code == 404


def test_download_404(c):
    client, _ = c
    assert client.get("/api/download/nope").status_code == 404


def test_index_serves_ui(c):
    client, _ = c
    r = client.get("/")
    assert r.status_code == 200
    assert "Model Squeezer" in r.text
