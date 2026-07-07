"""Tests for API auth and rate limiting."""

import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from textSummarizer.serving.app import app


@pytest.mark.asyncio
async def test_train_requires_key_when_enforced():
    with (
        patch.dict(os.environ, {"REQUIRE_TRAIN_KEY": "1", "TRAIN_API_KEY": "secret"}),
        patch("textSummarizer.serving.app.DataIngestionTrainingPipeline") as mock_ingest,
        patch("textSummarizer.serving.app.DataValidationTrainingPipeline") as mock_validate,
        patch("textSummarizer.serving.app.DataTransformationTrainingPipeline") as mock_transform,
        patch("textSummarizer.serving.app.ModelTrainerTrainingPipeline") as mock_train,
        patch("textSummarizer.serving.app.ModelEvaluationTrainingPipeline") as mock_eval,
    ):
        for mock_stage in (mock_ingest, mock_validate, mock_transform, mock_train, mock_eval):
            mock_stage.return_value.main.return_value = None
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/train")
        assert response.status_code == 401

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/train", headers={"X-API-Key": "secret"})
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_train_fails_closed_without_key_configured():
    with patch.dict(os.environ, {"REQUIRE_TRAIN_KEY": "1"}, clear=False):
        env = os.environ.copy()
        env.pop("TRAIN_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/train", headers={"X-API-Key": "anything"})
            assert response.status_code == 503


@pytest.mark.asyncio
async def test_api_key_required_when_configured():
    with patch.dict(os.environ, {"API_KEY": "test-key"}):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/summarize",
                json={
                    "text": "AI is changing the world.",
                    "model": "extractive",
                    "strategy": "stuff",
                    "max_length": 64,
                },
            )
        assert response.status_code == 401

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/summarize",
                json={
                    "text": "AI is changing the world.",
                    "model": "extractive",
                    "strategy": "stuff",
                    "max_length": 64,
                },
                headers={"X-API-Key": "test-key"},
            )
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_summarize_stream():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize/stream",
            json={
                "text": "AI transforms industries.",
                "model": "extractive",
                "strategy": "stuff",
                "max_length": 64,
            },
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "data:" in response.text


@pytest.mark.asyncio
async def test_summarize_with_citations_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize/citations",
            json={
                "text": "AI transforms healthcare. ML detects diseases.",
                "model": "extractive",
                "strategy": "stuff",
                "max_length": 64,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]
    assert "citations" in data
