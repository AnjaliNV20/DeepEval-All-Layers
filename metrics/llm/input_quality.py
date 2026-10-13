import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


COMMON_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "does",
    "explain",
    "for",
    "how",
    "is",
    "of",
    "please",
    "tell",
    "the",
    "to",
    "what",
    "where",
    "why",
}


def _words(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


class SpellingMistakeMetric(BaseMetric):
    def __init__(self, reference_texts, threshold=0.0):
        self.reference_texts = reference_texts
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self):
        return "Spelling Mistake Score"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        vocabulary = set(COMMON_WORDS)
        for reference in self.reference_texts:
            vocabulary.update(_words(reference))

        prompt_words = _words(test_case.input)
        misspelled_words = sorted({word for word in prompt_words if word not in vocabulary})
        correct_word_count = len(prompt_words) - len(misspelled_words)
        self.score = correct_word_count / len(prompt_words) if prompt_words else 1.0
        self.success = self.is_successful()

        if misspelled_words:
            self.reason = (
                f"Detected possible spelling mistakes: {', '.join(misspelled_words)}. "
                f"Prompt spelling score: {self.score:.2f}."
            )
        else:
            self.reason = f"No spelling mistakes detected. Prompt spelling score: {self.score:.2f}."

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)


class RobustnessMetric(BaseMetric):
    def __init__(self, spelling_metric, threshold=0.8):
        self.spelling_metric = spelling_metric
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self):
        return "Spelling Robustness"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        spelling_score = self.spelling_metric.measure(test_case)
        expected_words = set(_words(test_case.expected_output or ""))
        actual_words = set(_words(test_case.actual_output or ""))
        overlap = (
            len(expected_words & actual_words) / len(expected_words)
            if expected_words
            else 1.0
        )
        self.score = overlap
        self.success = self.is_successful()

        if spelling_score < 1.0:
            self.reason = (
                f"The prompt contained spelling mistakes, and the answer preserved "
                f"{overlap:.2f} of the expected meaning."
            )
        else:
            self.reason = "The prompt had no detected spelling mistakes."

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)