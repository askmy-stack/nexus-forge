import logging

logger = logging.getLogger(__name__)


class EvaluationSuite:
    """Multi-metric evaluation with tiered cost."""

    TIER_METRICS = {
        1: ["rouge"],
        2: ["rouge", "bertscore"],
        3: ["rouge", "bertscore", "summac"],
        4: ["rouge", "bertscore", "summac", "geval"],
    }

    def __init__(self, tier: int = 1, use_geval: bool | None = None):
        if tier not in self.TIER_METRICS:
            raise ValueError(f"Invalid tier {tier}. Choose from {list(self.TIER_METRICS)}")
        self.tier = tier
        self.metric_names = self.TIER_METRICS[tier]
        self.use_geval = use_geval
        self._metrics: dict = {}

    def _load_metric(self, name: str):
        if name in self._metrics:
            return self._metrics[name]
        import evaluate

        metric = evaluate.load(name)
        self._metrics[name] = metric
        return metric

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        sources: list[str] | None = None,
    ) -> dict[str, float]:
        results: dict[str, float] = {}

        if "rouge" in self.metric_names:
            rouge = self._load_metric("rouge")
            rouge_scores = rouge.compute(predictions=predictions, references=references)
            for key, value in rouge_scores.items():
                results[f"rouge_{key}"] = float(value)

        if "bertscore" in self.metric_names:
            try:
                bertscore = self._load_metric("bertscore")
                bs = bertscore.compute(
                    predictions=predictions,
                    references=references,
                    lang="en",
                )
                results["bertscore_f1"] = float(sum(bs["f1"]) / len(bs["f1"]))
            except Exception as exc:
                logger.warning(f"BERTScore unavailable: {exc}")

        if "summac" in self.metric_names and sources:
            try:
                from summac.model_summac import SummaCConv

                summac = SummaCConv(models=["vitc"], bins="percentile", granularity="sentence")
                score = summac.score(sources, predictions)
                results["summac_conv"] = float(score["scores"][0])
            except Exception as exc:
                logger.warning(f"SummaC unavailable: {exc}")

        if "geval" in self.metric_names and sources:
            from textSummarizer.grading.geval import geval_score, is_geval_available

            geval_results = [
                geval_score(source, prediction, use_geval=self.use_geval)
                for source, prediction in zip(sources, predictions, strict=True)
            ]
            geval_scores = [float(result["geval_score"]) for result in geval_results]
            results["geval_score"] = float(sum(geval_scores) / len(geval_scores))
            methods = {str(result.get("method", "unknown")) for result in geval_results}
            results["geval_method"] = "deepeval" if methods == {"deepeval"} else next(iter(methods))
            if is_geval_available() and self.use_geval is not False:
                logger.info("Tier %s G-Eval used deepeval LLM judge", self.tier)

        return results
