from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_type: str  # abstractive | extractive
    max_input_tokens: int
    requires_prefix: str | None = None
    description: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "pegasus": ModelSpec(
        model_id="google/pegasus-cnn_dailymail",
        model_type="abstractive",
        max_input_tokens=1024,
        description="Pegasus fine-tuned on CNN/DailyMail",
    ),
    "pegasus-xsum": ModelSpec(
        model_id="google/pegasus-xsum",
        model_type="abstractive",
        max_input_tokens=1024,
        description="Pegasus fine-tuned on XSum (extreme abstractive)",
    ),
    "bart": ModelSpec(
        model_id="facebook/bart-large-cnn",
        model_type="abstractive",
        max_input_tokens=1024,
        description="BART large CNN/DailyMail",
    ),
    "t5": ModelSpec(
        model_id="google-t5/t5-base",
        model_type="abstractive",
        max_input_tokens=512,
        requires_prefix="summarize: ",
        description="T5 base for summarization",
    ),
    "flan-t5": ModelSpec(
        model_id="google/flan-t5-base",
        model_type="abstractive",
        max_input_tokens=512,
        requires_prefix="summarize: ",
        description="FLAN-T5 instruction-tuned",
    ),
    "longt5": ModelSpec(
        model_id="google/long-t5-tglobal-base",
        model_type="abstractive",
        max_input_tokens=16384,
        description="LongT5 for long documents",
    ),
    "extractive": ModelSpec(
        model_id="extractive-textrank",
        model_type="extractive",
        max_input_tokens=10000,
        description="Extractive summarization via TextRank sentence ranking",
    ),
}
