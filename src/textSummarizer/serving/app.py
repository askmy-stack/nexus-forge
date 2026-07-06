import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.models import ModelFactory
from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    model: str = Field(default="extractive", description="Model name from registry")
    strategy: str = Field(default="stuff", pattern="^(stuff|map_reduce|refine)$")
    max_length: int = Field(default=128, ge=16, le=512)


class SummarizeResponse(BaseModel):
    summary: str
    model: str
    strategy: str


def _verify_train_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("TRAIN_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelFactory.list_models()
    yield


app = FastAPI(
    title="Text Summarization API",
    description="Extractive and abstractive summarization with multiple transformer models",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["docs"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/models", tags=["models"])
async def list_models():
    return ModelFactory.list_models()


@app.post("/summarize", response_model=SummarizeResponse, tags=["inference"])
async def summarize(request: SummarizeRequest):
    if request.model == "pegasus":
        pipeline = PredictionPipeline(model_name="pegasus")
        summary = pipeline.predict(
            request.text, strategy=request.strategy, max_length=request.max_length
        )
    elif request.model == "extractive":
        summarizer = ModelFactory.create("extractive")
        summary = summarizer.summarize(request.text, max_length=request.max_length)
    else:
        summarizer = ModelFactory.create(request.model)
        summary = summarizer.summarize(
            request.text, max_length=request.max_length, strategy=request.strategy
        )

    return SummarizeResponse(summary=summary, model=request.model, strategy=request.strategy)


@app.post("/train", tags=["training"], dependencies=[Depends(_verify_train_key)])
async def train_pipeline():
    stages = [
        DataIngestionTrainingPipeline(),
        DataValidationTrainingPipeline(),
        DataTransformationTrainingPipeline(),
        ModelTrainerTrainingPipeline(),
        ModelEvaluationTrainingPipeline(),
    ]
    for stage in stages:
        stage.main()
    return {"status": "training completed"}
