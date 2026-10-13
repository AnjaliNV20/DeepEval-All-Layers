"""Name -> metric lookups used by the JSON-driven test suites.

Test data files list metrics by name (for example "Faithfulness" or
"BiasMetric"). These helpers turn those names into ready-to-use metric
objects, so any test can reuse the same metrics.
"""

from .llm.faithfulness import get_faithfulness_metric
from .llm.hallucination import get_hallucination_metric
from .llm.correctness import get_correctness_metric
from .llm.completeness import get_completeness_metric
from .llm.clarity import get_clarity_metric
from .llm.groundedness import get_groundedness_metric
from .llm.robustness_geval import get_robustness_geval_metric
from .llm.prompt_alignment_geval import get_prompt_alignment_geval_metric
from .llm.geval_general import get_geval_metric
from .llm.tone import get_tone_metric
from .llm.policy_compliance import get_policy_compliance_metric
from .llm.disclaimer_presence import get_disclaimer_presence_metric
from .safety.toxicity import get_toxicity_metric
from .safety.bias import get_bias_metric
from .safety.prompt_injection import get_prompt_injection_metric
from .safety.jailbreak import get_jailbreak_metric
from .ui.answer_relevancy import get_answer_relevancy_metric
from .ui.conversation_relevancy import get_conversation_relevancy_metric


def _named(builder, name):
    """Build a metric and give it the display name the test reports use."""

    def build(model=None):
        metric = builder(model)
        metric.name = name
        return metric

    return build


# ---------------- LLM layer ----------------
LLM_METRICS = {
    "Correctness": get_correctness_metric,
    "Completeness": get_completeness_metric,
    "Clarity": get_clarity_metric,
    "Faithfulness": _named(get_faithfulness_metric, "Faithfulness"),
    "Groundedness": get_groundedness_metric,
    "Robustness": get_robustness_geval_metric,
    "Prompt Alignment": get_prompt_alignment_geval_metric,
    "Hallucination Detection": _named(get_hallucination_metric, "HallucinationMetric"),
    "HallucinationMetric": _named(get_hallucination_metric, "HallucinationMetric"),
    "Toxicity": _named(get_toxicity_metric, "ToxicityMetric"),
    "ToxicityMetric": _named(get_toxicity_metric, "ToxicityMetric"),
    "Bias": _named(get_bias_metric, "BiasMetric"),
    "BiasMetric": _named(get_bias_metric, "BiasMetric"),
    "GEval": get_geval_metric,
    "Prompt Injection Resistance": get_prompt_injection_metric,
    "Jailbreak Resistance": get_jailbreak_metric,
}

DEFAULT_LLM_METRICS = [
    "Correctness",
    "Completeness",
    "HallucinationMetric",
    "ToxicityMetric",
    "BiasMetric",
    "GEval",
]


def build_llm_metric_suite(testcase=None, model=None):
    names = testcase.get("llm_metrics", []) if testcase else []
    if not names:
        names = DEFAULT_LLM_METRICS
    return [LLM_METRICS[name](model=model) for name in names]


# ---------------- Safety (bias + toxicity) ----------------
def build_safety_metric_suite(testcase=None, model=None):
    names = testcase.get("llm_metrics", []) if testcase else []

    selected = []
    for name in names:
        if name in {"Bias", "BiasMetric"}:
            selected.append("BiasMetric")
        elif name in {"Toxicity", "ToxicityMetric"}:
            selected.append("ToxicityMetric")

    if not selected:
        selected = ["BiasMetric", "ToxicityMetric"]

    selected = set(selected)
    ordered = [name for name in ["BiasMetric", "ToxicityMetric"] if name in selected]
    return [LLM_METRICS[name](model=model) for name in ordered]


# ---------------- Custom G-Eval rubrics ----------------
CUSTOM_METRICS = {
    "Tone": get_tone_metric,
    "Policy Compliance": get_policy_compliance_metric,
    "Disclaimer Presence": get_disclaimer_presence_metric,
}


def build_custom_metric_suite(testcase=None, model=None):
    names = testcase.get("custom_metrics", []) if testcase else []
    return [CUSTOM_METRICS[name](model=model) for name in dict.fromkeys(names)]


# ---------------- UI layer ----------------
def build_ui_metric_suite(testcase=None, model=None, conversation_criteria=None):
    names = testcase.get("ui_metrics", []) if testcase else []
    if not names:
        names = ["Answer Relevancy", "Conversation Relevancy"]

    def conversation_relevancy(model=None):
        if conversation_criteria is None:
            return get_conversation_relevancy_metric(model)
        return get_conversation_relevancy_metric(model, criteria=conversation_criteria)

    ui_metrics = {
        "Answer Relevancy": get_answer_relevancy_metric,
        "Conversation Relevancy": conversation_relevancy,
    }
    return [ui_metrics[name](model=model) for name in names]
