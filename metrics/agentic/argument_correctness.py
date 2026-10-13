from deepeval.metrics import ArgumentCorrectnessMetric


def get_argument_correctness_metric(model=None):
    return ArgumentCorrectnessMetric(
        threshold=0.5,
        model=model,
    )
