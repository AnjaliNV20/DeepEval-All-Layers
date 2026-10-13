from deepeval.metrics import AnswerRelevancyMetric, ConversationalGEval
from deepeval.test_case import MultiTurnParams


def build_answer_relevancy_metric(model=None):
    return AnswerRelevancyMetric(
        model=model,
        threshold=0.7,
    )


def build_conversation_relevancy_metric(model=None):
    return ConversationalGEval(
        name="Conversation Relevancy",
        criteria="The assistant should remain relevant to the user's intent throughout the conversation.",
        evaluation_params=[MultiTurnParams.CONTENT],
        model=model,
        threshold=0.7,
    )


def build_ui_metric_suite(testcase=None, model=None):
    metrics = testcase.get("ui_metrics", []) if testcase else []

    if not metrics:
        metrics = [
            "Answer Relevancy",
            "Conversation Relevancy",
        ]

    metric_map = {
        "Answer Relevancy": build_answer_relevancy_metric,
        "Conversation Relevancy": build_conversation_relevancy_metric,
    }

    return [
        metric_map[name](model=model)
        for name in metrics
    ]