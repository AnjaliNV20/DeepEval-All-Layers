from deepeval.metrics import HallucinationMetric
 
 
def get_hallucination_metric(model):
    return HallucinationMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )