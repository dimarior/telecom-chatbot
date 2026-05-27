"""
tests/test_api.py
─────────────────
Tests básicos de la API FastAPI.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def test_health_endpoint():
    """Test endpoint /health."""
    with patch("src.config.VECTORSTORE_DIR") as mock_vs:
        from api.main import app
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
