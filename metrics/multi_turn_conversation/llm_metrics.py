from deepeval.metrics import FaithfulnessMetric


def build_faithfulness_metric(model=None):
    return FaithfulnessMetric(
        model=model,
        threshold=0.5,
    )


def build_llm_metric_suite(testcase=None, model=None):
    metrics = testcase.get("llm_metrics", []) if testcase else []

    if not metrics:
        metrics = [
            "Faithfulness"
        ]

    metric_map = {
        "Faithfulness": build_faithfulness_metric,
    }

    return [
        metric_map[name](model=model)
        for name in metrics
    ]