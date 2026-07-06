import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import RedirectResponse

from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.grading.llm_judge import LLMJudge
from textSummarizer.grading.rubric import GradingRubric
from textSummarizer.models import ModelFactory
from textSummarizer.multimodal.base import InputType, MultimodalInput
from textSummarizer.multimodal.router import MultimodalRouter
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


class MultimodalJsonRequest(BaseModel):
    input_type: Literal["text", "image", "audio", "video"] = Field(
        ..., description="Modality of the input"
    )
    text: str | None = Field(default=None, description="Plain text (for text modality)")
    base64_data: str | None = Field(
        default=None, description="Base64-encoded file content (image/audio)"
    )
    path: str | None = Field(default=None, description="Server-side file path")
    model: str = Field(default="extractive", description="Text summarization model")
    strategy: str = Field(
        default="stuff",
        pattern="^(stuff|map_reduce|refine)$",
        description="Long-document strategy",
    )
    max_length: int = Field(default=128, ge=16, le=512, description="Maximum summary length")


class MultimodalResponse(BaseModel):
    input_type: str
    summary: str
    model: str
    strategy: str
    caption: str | None = None
    transcript: str | None = None


class GradeRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=50000, description="Source text")
    summary: str = Field(..., min_length=1, max_length=10000, description="Summary to grade")
    threshold: float = Field(default=3.5, ge=1.0, le=5.0, description="Pass threshold (1-5)")


class GradeScoreResponse(BaseModel):
    coherence: int
    faithfulness: int
    fluency: int
    relevance: int
    average: float
    feedback: str


class GradeResponse(BaseModel):
    score: GradeScoreResponse
    passes: bool
    threshold: float


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
    title="SummarizeHub API",
    description=(
        "Multimodal summarization platform — text, image, and audio inputs with "
        "subjective grading and MCP agent integration"
    ),
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


def _multimodal_result_to_response(result: dict, strategy: str) -> MultimodalResponse:
    return MultimodalResponse(
        input_type=result["input_type"],
        summary=result["summary"],
        model=result["model"],
        strategy=strategy,
        caption=result.get("caption"),
        transcript=result.get("transcript"),
    )


@app.post("/summarize/multimodal", response_model=MultimodalResponse, tags=["inference"])
async def summarize_multimodal_json(request: MultimodalJsonRequest):
    try:
        input_type = InputType(request.input_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input_type: {request.input_type}",
        ) from exc

    router = MultimodalRouter(text_model=request.model)
    try:
        result = router.summarize(
            MultimodalInput(
                input_type=input_type,
                text=request.text,
                base64_data=request.base64_data,
                path=request.path,
            ),
            max_length=request.max_length,
            strategy=request.strategy,
        )
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _multimodal_result_to_response(result, request.strategy)


@app.post(
    "/summarize/multimodal/upload",
    response_model=MultimodalResponse,
    tags=["inference"],
)
async def summarize_multimodal_upload(
    file: Annotated[UploadFile, File()],
    input_type: Literal["image", "audio"] = Form(...),
    model: str = Form(default="extractive"),
    strategy: str = Form(default="stuff"),
    max_length: int = Form(default=128, ge=16, le=512),
):
    content = await file.read()
    router = MultimodalRouter(text_model=model)
    try:
        result = router.summarize(
            MultimodalInput(
                input_type=InputType(input_type),
                data=content,
                metadata={"filename": file.filename or ""},
            ),
            max_length=max_length,
            strategy=strategy,
        )
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _multimodal_result_to_response(result, strategy)


@app.post("/grade", response_model=GradeResponse, tags=["grading"])
async def grade_summary(request: GradeRequest):
    rubric = GradingRubric(threshold=request.threshold)
    judge = LLMJudge(use_llm=False)
    score = judge.grade(request.source, request.summary)
    return GradeResponse(
        score=GradeScoreResponse(**score.to_dict()),
        passes=rubric.passes(score),
        threshold=request.threshold,
    )


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
