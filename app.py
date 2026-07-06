"""Entry point for Docker and local uvicorn."""

from textSummarizer.serving.app import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
