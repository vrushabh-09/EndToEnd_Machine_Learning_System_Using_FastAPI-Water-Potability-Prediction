import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


VALID_SAMPLE = {
    "ph": 7.1,
    "Hardness": 196.9,
    "Solids": 20927.8,
    "Chloramines": 7.1,
    "Sulfate": 333.1,
    "Conductivity": 421.9,
    "Organic_carbon": 14.2,
    "Trihalomethanes": 66.6,
    "Turbidity": 3.9,
}


def test_root_serves_something(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_sample(client):
    resp = client.post("/predict", json=VALID_SAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert "potable" in body
    assert body["label"] in ("Potable", "Not potable")
    assert 0.0 <= body["probability_potable"] <= 1.0
    assert body["confidence"] in ("low", "medium", "high")


def test_predict_partial_sample_is_imputed(client):
    """Model pipeline has a median imputer, so a partial sample should still work."""
    partial = {"ph": 7.0, "Hardness": 200.0}
    resp = client.post("/predict", json=partial)
    assert resp.status_code == 200


def test_predict_empty_body_rejected(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_out_of_range_ph_rejected(client):
    bad = dict(VALID_SAMPLE)
    bad["ph"] = 40.0  # pH only goes 0-14
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_negative_value_rejected(client):
    bad = dict(VALID_SAMPLE)
    bad["Solids"] = -100.0
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_wrong_type_rejected(client):
    bad = dict(VALID_SAMPLE)
    bad["ph"] = "not-a-number"
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


@pytest.mark.parametrize("field", list(VALID_SAMPLE.keys()))
def test_each_field_individually_out_of_bounds(client, field):
    bad = dict(VALID_SAMPLE)
    bad[field] = -1  # every field has ge=0
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
