import json
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.entity import ModelEvaluationConfig
from textSummarizer.evaluation.metrics import EvaluationSuite
from textSummarizer.logging import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    @staticmethod
    def generate_batch_sized_chunks(items: list, batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric_suite: EvaluationSuite,
        model,
        tokenizer,
        batch_size: int = 2,
        device: str | None = None,
        column_text: str = "dialogue",
        column_summary: str = "summary",
    ) -> dict:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        all_predictions: list[str] = []
        all_references: list[str] = []
        all_sources: list[str] = []

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches, strict=False),
            total=len(article_batches),
        ):
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128,
            )
            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for s in summaries
            ]
            all_predictions.extend(decoded_summaries)
            all_references.extend(target_batch)
            all_sources.extend(article_batch)

        return metric_suite.evaluate(all_predictions, all_references, all_sources)

    def evaluate(self) -> dict:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_path)).to(device)

        dataset_samsum_pt = load_from_disk(str(self.config.data_path))
        eval_subset = dataset_samsum_pt["test"].select(
            range(min(self.config.eval_samples, len(dataset_samsum_pt["test"])))
        )

        metric_suite = EvaluationSuite(tier=1)
        scores = self.calculate_metric_on_test_ds(
            eval_subset,
            metric_suite,
            model,
            tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary",
        )

        output = {"model": "pegasus-samsum", "metrics": scores}
        Path(self.config.metric_file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.metric_file_name, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Evaluation metrics saved to {self.config.metric_file_name}")
        return scores
