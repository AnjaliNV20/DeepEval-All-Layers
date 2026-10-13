import ollama

from deepeval.test_case import ConversationalTestCase, Turn

from config.models import (
    get_answer_model_name,
    get_judge_model,
)

from metrics import (
    build_ui_metric_suite,
    CONVERSATION_RELEVANCY_ORANGEHRM_CRITERIA,
)
from utils.testcase_loader import get_test_case


TEST_CASE_IDS = [
    "TC_SC2_P1",
    "TC_SC2_P2",
    "TC_SC2_P3",
    "TC_SC2_P4",
    "TC_SC2_P5",
]


def test_conversation_relevancy():

    # Complete LLM conversation history
    conversation = []

    # DeepEval conversation turns
    turns = []

    # Run P1 → P5 as one continuous conversation
    for testcase_id in TEST_CASE_IDS:

        # Get current prompt from JSON
        testcase = get_test_case("multi_turn_conversation", testcase_id)
        question = testcase["prompt"]

        # Send previous conversation + current question to Answer LLM
        response = ollama.chat(
            model=get_answer_model_name(),
            messages=conversation + [
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        # Get LLM response
        answer = response.message.content

        # Add user question to conversation history
        conversation.append({
            "role": "user",
            "content": question
        })

        # Add AI answer to conversation history
        conversation.append({
            "role": "assistant",
            "content": answer
        })

        # Add user turn for DeepEval
        turns.append(
            Turn(
                role="user",
                content=question
            )
        )

        # Add assistant turn for DeepEval
        turns.append(
            Turn(
                role="assistant",
                content=answer
            )
        )

    # Create complete conversation test case
    test_case = ConversationalTestCase(
        turns=turns
    )

    # Get Judge LLM
    judge_model = get_judge_model()

    # Build UI metric
    metrics = build_ui_metric_suite(
        testcase={
            "ui_metrics": ["Conversation Relevancy"]
        },
        model=judge_model,
        conversation_criteria=CONVERSATION_RELEVANCY_ORANGEHRM_CRITERIA
    )

    # Evaluate conversation
    for metric in metrics:

        metric.measure(test_case)

        print("\n================================")
        print("SCENARIO 2 - UI LAYER")
        print("================================")

        print("\nConversation:")

        for message in conversation:
            print(
                message["role"],
                ":",
                message["content"]
            )

        print("\nMetric:", metric.__class__.__name__)
        print("Score:", metric.score)

        print("================================")

        assert metric.score >= 0.7