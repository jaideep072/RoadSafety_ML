"""
Integration tests for Flask web routes, Jinja templates, and JSON REST API endpoints.
"""

import pytest
from app import create_app
from app.config import Config


@pytest.fixture
def client():
    app = create_app(Config)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Road Safety" in response.data


def test_data_loading_route(client):
    response = client.get("/data-loading")
    assert response.status_code == 200


def test_eda_route(client):
    response = client.get("/eda")
    assert response.status_code == 200


def test_preprocessing_route(client):
    response = client.get("/preprocessing")
    assert response.status_code == 200


def test_linear_regression_routes(client):
    # GET
    res_get = client.get("/linear-regression?reg_type=ridge")
    assert res_get.status_code == 200

    # POST
    payload = {
        "Temperature(F)": "75.0",
        "Humidity(%)": "50.0",
        "Pressure(in)": "29.92",
        "Visibility(mi)": "10.0",
        "Wind_Speed(mph)": "5.0",
        "Crossing": "0",
        "Junction": "1",
        "Traffic_Signal": "0"
    }
    res_post = client.post("/linear-regression?reg_type=none", data=payload)
    assert res_post.status_code == 200


def test_logistic_regression_routes(client):
    # GET
    res_get = client.get("/logistic-regression?reg_type=lasso")
    assert res_get.status_code == 200

    # POST
    payload = {
        "Temperature(F)": "85.0",
        "Humidity(%)": "70.0",
        "Pressure(in)": "29.80",
        "Visibility(mi)": "4.0",
        "Wind_Speed(mph)": "15.0",
        "Crossing": "1",
        "Junction": "1",
        "Traffic_Signal": "1",
        "Sunrise_Sunset_Day": "0"
    }
    res_post = client.post("/logistic-regression?reg_type=none", data=payload)
    assert res_post.status_code == 200


def test_decision_trees_routes(client):
    res_get = client.get("/decision-trees?algo=rf")
    assert res_get.status_code == 200


def test_clustering_routes(client):
    res_kmeans = client.get("/clustering?algo=kmeans")
    assert res_kmeans.status_code == 200

    res_dbscan = client.get("/clustering?algo=dbscan")
    assert res_dbscan.status_code == 200

    res_hierarchical = client.get("/clustering?algo=hierarchical")
    assert res_hierarchical.status_code == 200

    res_geo = client.get("/clustering?algo=geographic")
    assert res_geo.status_code == 200


def test_api_endpoints(client):
    # Data summary API
    res_sum = client.get("/api/data-summary")
    assert res_sum.status_code == 200
    assert res_sum.json["status"] == "success"

    # EDA API
    res_eda = client.get("/api/eda")
    assert res_eda.status_code == 200

    # Preprocessing API
    res_prep = client.get("/api/preprocessing")
    assert res_prep.status_code == 200

    # Prediction APIs
    pred_payload = {
        "model": "linear_reg",
        "features": {"Temperature(F)": 70, "Humidity(%)": 60, "Pressure(in)": 29.9, "Visibility(mi)": 10, "Wind_Speed(mph)": 5}
    }
    res_pred = client.post("/api/predict/linear", json=pred_payload)
    assert res_pred.status_code == 200
    assert res_pred.json["status"] == "success"

    # Clustering API
    res_km_api = client.post("/api/clustering/kmeans", json={"n_clusters": 3})
    assert res_km_api.status_code == 200
    assert res_km_api.json["status"] == "success"


def test_404_handling(client):
    response = client.get("/non-existent-page-url")
    assert response.status_code == 404
    assert response.json["status"] == "error"
