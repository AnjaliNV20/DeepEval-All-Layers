from deepeval.metrics import TaskCompletionMetric


def get_task_completion_metric(model=None):
    return TaskCompletionMetric(
        threshold=0.5,
        model=model,
    )
