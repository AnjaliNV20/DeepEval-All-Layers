from .safety_checks import HATE_SPEECH_PATTERNS
from .safety_checks import SafeRefusalMetric


def get_hate_speech_metric():
    return SafeRefusalMetric(
        name="Hate Speech Detection",
        unsafe_patterns=HATE_SPEECH_PATTERNS,
        threshold=0.8
    )