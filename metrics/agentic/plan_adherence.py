from deepeval.metrics import PlanAdherenceMetric


def get_plan_adherence_metric(model=None):
    return PlanAdherenceMetric(
        threshold=0.5,
        model=model,
    )
