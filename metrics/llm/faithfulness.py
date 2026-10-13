from deepeval.metrics import FaithfulnessMetric
 
 
def get_faithfulness_metric(model):
    return FaithfulnessMetric(
        threshold=0.5,
        model=model,
        include_reason=True
    )