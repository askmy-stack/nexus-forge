import pytest
from httpx import ASGITransport, AsyncClient

from textSummarizer.serving.app import app


@pytest.mark.asyncio
async def test_grade_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/grade",
            json={
                "source": "AI is reshaping healthcare. Machine learning detects disease.",
                "summary": "AI reshapes healthcare with machine learning.",
                "threshold": 3.0,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert data["score"]["average"] >= 1.0
    assert "passes" in data


@pytest.mark.asyncio
async def test_multimodal_text_json():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize/multimodal",
            json={
                "input_type": "text",
                "text": "AI is changing the world. Machine learning powers assistants.",
                "model": "extractive",
                "strategy": "stuff",
                "max_length": 64,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "text"
    assert data["summary"]


@pytest.mark.asyncio
async def test_multimodal_invalid_type():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize/multimodal",
            json={
                "input_type": "video",
                "path": "/tmp/video.mp4",
                "model": "extractive",
            },
        )
    assert response.status_code == 422
