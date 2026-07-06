import pytest
from httpx import ASGITransport, AsyncClient

from textSummarizer.serving.app import app


@pytest.mark.asyncio
async def test_list_models():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/models")
    assert response.status_code == 200
    assert "extractive" in response.json()


@pytest.mark.asyncio
async def test_summarize_extractive():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize",
            json={
                "text": "AI is changing the world. Machine learning powers assistants.",
                "model": "extractive",
                "strategy": "stuff",
                "max_length": 64,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]
    assert data["model"] == "extractive"
