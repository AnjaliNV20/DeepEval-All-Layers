from .safety_checks import ACTIONABLE_VIOLENCE_PATTERNS
from .safety_checks import SafeRefusalMetric


def get_violence_detection_metric(model):
    return SafeRefusalMetric(
        name="Violence Detection",
        unsafe_patterns=ACTIONABLE_VIOLENCE_PATTERNS,
        threshold=0.8
    )