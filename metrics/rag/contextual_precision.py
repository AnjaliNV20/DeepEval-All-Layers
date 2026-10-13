from deepeval.metrics import ContextualPrecisionMetric
from config.settings import THRESHOLD
def create_contextual_precision_metric(model):
    return ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=model,
        include_reason=True,
        async_mode=False
    )