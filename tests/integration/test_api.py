import pytest
from httpx import ASGITransport, AsyncClient

from textSummarizer.serving.app import app


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_available"] >= 1


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


@pytest.mark.asyncio
async def test_summarize_extractive_map_reduce():
    transport = ASGITransport(app=app)
    text = ". ".join([f"Sentence number {i} with useful content here" for i in range(25)])
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize",
            json={
                "text": text,
                "model": "extractive",
                "strategy": "map_reduce",
                "max_length": 64,
            },
        )
    assert response.status_code == 200
    assert response.json()["summary"]


@pytest.mark.asyncio
async def test_summarize_unknown_model_returns_422():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/summarize",
            json={
                "text": "Hello world.",
                "model": "not-a-real-model",
                "strategy": "stuff",
                "max_length": 64,
            },
        )
    assert response.status_code == 422
    assert "Unknown model" in response.json()["detail"]
