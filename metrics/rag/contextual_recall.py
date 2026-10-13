from deepeval.metrics import ContextualRecallMetric
from config.settings import THRESHOLD
def create_contextual_recall_metric(model):
    return ContextualRecallMetric(
        threshold=THRESHOLD,
        model=model,
        include_reason=True,
        async_mode=False
    )