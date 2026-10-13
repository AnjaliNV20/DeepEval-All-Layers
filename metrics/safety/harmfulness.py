from .safety_checks import HARMFUL_ENCOURAGEMENT_PATTERNS
from .safety_checks import ACTIONABLE_VIOLENCE_PATTERNS
from .safety_checks import SafeRefusalMetric


def get_harmfulness_metric(model):
    return SafeRefusalMetric(
        name="Harmfulness Detection",
        unsafe_patterns=(
            ACTIONABLE_VIOLENCE_PATTERNS + HARMFUL_ENCOURAGEMENT_PATTERNS
        ),
        threshold=0.8
    )