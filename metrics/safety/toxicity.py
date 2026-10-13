from deepeval.metrics import ToxicityMetric
 
 
def get_toxicity_metric(model):
    return ToxicityMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )