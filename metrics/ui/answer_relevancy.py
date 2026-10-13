from deepeval.metrics import AnswerRelevancyMetric
 
 
def get_answer_relevancy_metric(model):
    return AnswerRelevancyMetric(
        threshold=0.5,
        model=model,
        include_reason=True,
        # async_mode=True 
    )