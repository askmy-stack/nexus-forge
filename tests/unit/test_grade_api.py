"""Tests for grade API with G-Eval."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from textSummarizer.serving.app import app


@pytest.mark.asyncio
async def test_grade_without_geval():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/grade",
            json={
                "source": "AI transforms healthcare and finance.",
                "summary": "AI transforms healthcare.",
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert data["geval"] is None


@pytest.mark.asyncio
async def test_grade_with_geval():
    transport = ASGITransport(app=app)
    mock_geval = {
        "geval_score": 0.91,
        "geval_reason": "Accurate and concise.",
        "method": "deepeval",
        "model": "gpt-4o-mini",
    }
    with patch("textSummarizer.serving.app.geval_score", return_value=mock_geval):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/grade",
                json={
                    "source": "AI transforms healthcare and finance.",
                    "summary": "AI transforms healthcare.",
                    "use_geval": True,
                },
            )
    assert response.status_code == 200
    data = response.json()
    assert data["geval"]["geval_score"] == 0.91
    assert data["geval"]["method"] == "deepeval"
