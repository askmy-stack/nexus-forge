import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import RedirectResponse

from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.models import ModelFactory
from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

API_VERSION = "0.1.0"

SUMMARIZE_EXAMPLE = {
    "text": (
        "Artificial intelligence is reshaping healthcare, finance, and transportation. "
        "Machine learning models can detect diseases from medical images. "
        "Natural language processing powers chatbots and document summarization."
    ),
    "model": "extractive",
    "strategy": "stuff",
    "max_length": 128,
}

SUMMARIZE_RESPONSE_EXAMPLE = {
    "summary": (
        "Artificial intelligence is reshaping healthcare, finance, and transportation. "
        "Machine learning models can detect diseases from medical images."
    ),
    "model": "extractive",
    "strategy": "stuff",
}


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [SUMMARIZE_EXAMPLE]},
    )

    text: str = Field(..., min_length=1, max_length=50000, description="Source text to summarize")
    model: str = Field(default="extractive", description="Model name from the registry")
    strategy: str = Field(
        default="stuff",
        pattern="^(stuff|map_reduce|refine)$",
        description="Long-document strategy",
    )
    max_length: int = Field(default=128, ge=16, le=512, description="Maximum summary length")


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [SUMMARIZE_RESPONSE_EXAMPLE]},
    )

    summary: str
    model: str
    strategy: str


class HealthResponse(BaseModel):
    status: str
    version: str
    models_available: int


def _verify_train_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("TRAIN_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _run_summarization(request: SummarizeRequest) -> str:
    try:
        if request.model == "pegasus":
            pipeline = PredictionPipeline(model_name="pegasus")
            return pipeline.predict(
                request.text, strategy=request.strategy, max_length=request.max_length
            )

        summarizer = ModelFactory.create(request.model)
        return summarizer.summarize(
            request.text, max_length=request.max_length, strategy=request.strategy
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelFactory.list_models()
    yield


app = FastAPI(
    title="Text Summarization API",
    description="Extractive and abstractive summarization with multiple transformer models",
    version=API_VERSION,
    lifespan=lifespan,
)


@app.get("/", tags=["docs"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        models_available=len(ModelFactory.list_models()),
    )


@app.get("/models", tags=["models"])
async def list_models():
    return ModelFactory.list_models()


@app.post("/summarize", response_model=SummarizeResponse, tags=["inference"])
async def summarize(request: SummarizeRequest):
    summary = _run_summarization(request)
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
