"""All evaluation metrics, in one place.

Import anything you need straight from `metrics`, for example:

    from metrics import get_faithfulness_metric, get_tool_correctness_metric

Folders (by layer):
    rag/      retrieval metrics (contextual precision, recall, relevancy)
    ui/       answer relevancy, conversation relevancy
    llm/      answer-quality metrics (faithfulness, hallucination, correctness, ...)
    safety/   toxicity, bias, harmfulness, hate speech, violence, jailbreak, ...
    agentic/  tool correctness, task completion, plan metrics, ...
    registry  turns metric names from JSON test data into metric objects
"""

# ---------------- RAG layer ----------------
from .rag.contextual_precision import create_contextual_precision_metric
from .rag.contextual_recall import create_contextual_recall_metric
from .rag.contextual_relevancy import create_contextual_relevancy_metric

# ---------------- UI layer ----------------
from .ui.answer_relevancy import get_answer_relevancy_metric
from .ui.conversation_relevancy import (
    get_conversation_relevancy_metric,
    DEFAULT_CRITERIA as CONVERSATION_RELEVANCY_DEFAULT_CRITERIA,
    ORANGEHRM_CRITERIA as CONVERSATION_RELEVANCY_ORANGEHRM_CRITERIA,
)

# ---------------- LLM layer ----------------
from .llm.faithfulness import get_faithfulness_metric
from .llm.hallucination import get_hallucination_metric
from .llm.correctness import get_correctness_metric
from .llm.completeness import get_completeness_metric
from .llm.clarity import get_clarity_metric
from .llm.groundedness import get_groundedness_metric
from .llm.prompt_alignment import get_prompt_alignment_metric
from .llm.prompt_alignment_geval import get_prompt_alignment_geval_metric
from .llm.robustness_geval import get_robustness_geval_metric
from .llm.geval_general import get_geval_metric
from .llm.tone import get_tone_metric
from .llm.policy_compliance import get_policy_compliance_metric
from .llm.disclaimer_presence import get_disclaimer_presence_metric
from .llm.input_quality import SpellingMistakeMetric, RobustnessMetric

# ---------------- Safety ----------------
from .safety.toxicity import get_toxicity_metric
from .safety.bias import get_bias_metric
from .safety.harmfulness import get_harmfulness_metric
from .safety.violence_detection import get_violence_detection_metric
from .safety.hate_speech import get_hate_speech_metric
from .safety.unsafe_content_rate import UnsafeContentRateMetric
from .safety.prompt_injection import get_prompt_injection_metric
from .safety.jailbreak import get_jailbreak_metric

# ---------------- Agentic / tool layer ----------------
from .agentic.tool_correctness import get_tool_correctness_metric
from .agentic.task_completion import get_task_completion_metric
from .agentic.argument_correctness import get_argument_correctness_metric
from .agentic.plan_adherence import get_plan_adherence_metric
from .agentic.plan_quality import get_plan_quality_metric

# ---------------- Name -> metric lookups for JSON-driven tests ----------------
from .registry import (
    build_llm_metric_suite,
    build_safety_metric_suite,
    build_custom_metric_suite,
    build_ui_metric_suite,
)
