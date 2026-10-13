from deepeval.metrics import BiasMetric
 
 
def get_bias_metric(model):
    return BiasMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )