from fastapi.testclient import TestClient

from API.api import app


client = TestClient(app)


def test_data_catalog_lists_main_dataset() -> None:
    response = client.get("/data/datasets")
    assert response.status_code == 200
    body = response.json()
    main = next(item for item in body["datasets"] if item["id"] == "wide_gold_full")
    assert main["rows"] == 405
    assert main["columns"] == 32
    assert "MAR" in main["countries"]


def test_data_rows_support_filters_and_pagination() -> None:
    response = client.get("/data/wide_gold_full?country=MAR&year_start=2020&year_end=2024&limit=2")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert len(body["rows"]) == 2
    assert {row["country_iso3"] for row in body["rows"]} == {"MAR"}


def test_data_unknown_dataset_returns_not_found() -> None:
    assert client.get("/data/not_a_dataset").status_code == 404


def test_data_analysis_returns_exploratory_aggregates() -> None:
    response = client.get("/data/analysis?country=MAR&year_start=2020&year_end=2024")
    assert response.status_code == 200
    body = response.json()
    assert body["scope"]["rows"] == 5
    assert len(body["summary"]) == 7
    assert len(body["country_summary"]) == 1
    assert len(body["correlations"]) == 7
    assert len(body["coverage"]) == 5