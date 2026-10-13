from deepeval.metrics import ContextualRelevancyMetric
from config.settings import THRESHOLD
def create_contextual_relevancy_metric(model):
    return ContextualRelevancyMetric(
        threshold=THRESHOLD,
        model=model,
        include_reason=True,
        async_mode=False
    )