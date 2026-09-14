from fastapi.testclient import TestClient

from API.api import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_expose_saved_artifacts() -> None:
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 35
    assert {item["target"] for item in body["models"]} == {
        "CROISSANCE_PIB",
        "INFLATION_CPI",
        "SOBG",
        "DETTE_PUBLIQUE",
        "BALANCE_COURANTE",
        "TACH",
        "FBCF",
    }


def test_predict_rejects_unknown_target() -> None:
    response = client.post(
        "/predict",
        json={"target": "INCONNUE", "horizon": 1, "history": [{"year": 2024, "values": {"CROISSANCE_PIB": 3.0}}] * 6},
    )
    assert response.status_code == 422


def test_predict_applies_saved_model() -> None:
    history = [
        {
            "year": year,
            "values": {
                "CROISSANCE_PIB": 2.0 + (year - 2019) * 0.1,
                "FBCF": 28.0 + (year - 2019) * 0.2,
                "IDEE": 1.5 + (year - 2019) * 0.05,
            },
        }
        for year in range(2019, 2025)
    ]
    response = client.post(
        "/predict",
        json={"target": "CROISSANCE_PIB", "horizon": 1, "history": history},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["origin_year"] == 2024
    assert body["target_year"] == 2025
    assert isinstance(body["prediction"], float)