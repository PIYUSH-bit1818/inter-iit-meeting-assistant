from fastapi.testclient import TestClient

from app.jobs import JobStore
from app.main import app

client = TestClient(app)


def test_health_ok_and_no_secret_values():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["config"]) == {
        "groq_api_key", "gemini_api_key", "stt_model", "refine_model", "document_model",
    }
    assert all(isinstance(v, bool) for v in body["config"].values())


def test_job_store_returns_copies():
    store = JobStore()
    job = store.create("a.wav")
    fetched = store.get(job.id)
    fetched.error = "mutated"
    assert store.get(job.id).error is None
    assert store.get("missing") is None
