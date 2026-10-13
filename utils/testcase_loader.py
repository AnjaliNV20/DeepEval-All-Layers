import json
from pathlib import Path

from config.settings import (
    AGENTIC_DATASET,
    MULTI_TURN_DATASET,
    SINGLE_TURN_DATASET,
)

# Test suites whose JSON file is a dict of {testcase_id: testcase}
SUITES = {
    "single_turn_ui_llm": SINGLE_TURN_DATASET,
    "multi_turn_conversation": MULTI_TURN_DATASET,
    "agentic_testing": AGENTIC_DATASET,
}


def load_test_cases(suite: str) -> dict:
    path = Path(SUITES[suite])
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_test_case(suite: str, testcase_id: str) -> dict:
    return load_test_cases(suite)[testcase_id]
