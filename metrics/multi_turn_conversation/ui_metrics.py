from deepeval.metrics import ConversationalGEval
from deepeval.test_case import MultiTurnParams


def build_conversation_relevancy_metric(model=None):
    return ConversationalGEval(
        name="Conversation Relevancy",
        criteria=(
            "The assistant should remain relevant to the user's intent "
            "throughout the multi-turn conversation. It should correctly "
            "understand references to the OrangeHRM login automation, "
            "Page Object, test script, API validation, and GitHub Actions."
        ),
        evaluation_params=[
            MultiTurnParams.CONTENT
        ],
        model=model,
        threshold=0.7,
    )


def build_ui_metric_suite(testcase=None, model=None):
    metrics = testcase.get("ui_metrics", []) if testcase else []

    if not metrics:
        metrics = [
            "Conversation Relevancy"
        ]

    metric_map = {
        "Conversation Relevancy": build_conversation_relevancy_metric,
    }

    return [
        metric_map[name](model=model)
        for name in metrics
    ]