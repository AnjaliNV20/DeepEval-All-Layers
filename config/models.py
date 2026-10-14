"""Ollama models used by the different test suites."""

from deepeval.models import OllamaModel

from config.settings import (
    ANSWER_MODEL_NAME,
    JUDGE_MODEL_NAME,
    OLLAMA_BASE_URL,
    TEMPERATURE,
    UI_LLM_MODEL,
)


def get_evaluation_model():
    """Judge model for the LLM / UI / safety tests."""
    return OllamaModel(
        model=UI_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )


def get_answer_model():
    """Answer-generation model for UI tests."""
    return OllamaModel(
        model=ANSWER_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )


def get_answer_model_name():
    """Name of the model that answers in the single-turn, multi-turn and agentic tests."""
    return ANSWER_MODEL_NAME


def get_judge_model():
    """Judge model for the single-turn, multi-turn and agentic tests."""
    return OllamaModel(
        model=JUDGE_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
    )
