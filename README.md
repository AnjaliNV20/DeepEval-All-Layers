# DeepEval Framework – All Layers of Conversational AI

One framework that evaluates every layer of a conversational AI system with
[DeepEval](https://deepeval.com), Ollama and Pytest.

## Layers covered

| Layer | What it checks | Metrics | Tests |
|---|---|---|---|
| RAG | Is the right context retrieved? | Contextual Precision, Recall, Relevancy | `tests/rag_evaluation` |
| UI + LLM + Safety | Answer quality, spelling robustness, harmful / hate-speech requests | Answer Relevancy, Faithfulness, Hallucination, Correctness, Prompt Alignment, Toxicity, Bias, Spelling Robustness, Harmfulness, Violence, Hate Speech, Unsafe Content Rate | `tests/llm_ui_safety` |
| Single-turn UI + LLM | Prompt in, answer out (38 scenarios) | Answer / Conversation Relevancy, Correctness, Completeness, Clarity, Groundedness, Robustness, Prompt Alignment, Faithfulness, Hallucination, Toxicity, Bias, GEval, Prompt Injection, Jailbreak, Tone, Policy Compliance, Disclaimer Presence | `tests/single_turn_ui_llm` |
| Multi-turn conversation | A 5-step OrangeHRM conversation | Conversation Relevancy, Faithfulness | `tests/multi_turn_conversation` |
| Agentic / tool | Does the agent pick the right tools and follow its plan? | Tool Correctness, Task Completion, Argument Correctness, Plan Adherence, Plan Quality | `tests/agentic_testing` |

## Folder structure

```
metrics/            ALL metrics live here (import from `metrics`)
  rag/  ui/  llm/  safety/  agentic/
  registry.py       turns metric names from JSON test data into metric objects
tests/              each suite has its own folder (see table above)
testdata/           each suite has its own JSON test data
  rag/  llm_ui_safety/  single_turn_ui_llm/  multi_turn_conversation/  agentic_testing/
config/
  settings.py       paths, models, thresholds (reads .env)
  models.py         Ollama model helpers
agents/             RAG agents + agentflow.py (tool-calling agent)
rag/                chunker, retriever, vector store
chatbot/            simple rule-based chatbot used by tests/llm_ui_safety
utils/              data loaders
scripts/            one-off helper scripts (not run by pytest)
documents/  vector_db/  build_vector_db.py
```

### Reusing a metric in any test

```python
from metrics import get_faithfulness_metric, get_toxicity_metric
from config.models import get_judge_model

metric = get_faithfulness_metric(get_judge_model())
metric.measure(test_case)
print(metric.score, metric.reason)
```

## Setup

```bash
python -m venv venv
# Windows:  .\venv\Scripts\Activate.ps1      Mac/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to change the models or Ollama address.

## Ollama

```bash
ollama pull qwen2.5:7b     # answering model
ollama pull qwen2.5:3b     # judge model for the single-turn / multi-turn / agentic tests
ollama list
```

## Run tests

```bash
pytest                                      # everything
pytest tests/rag_evaluation -v              # one layer
pytest tests/llm_ui_safety -v
pytest tests/single_turn_ui_llm -v
pytest tests/multi_turn_conversation -v
pytest tests/agentic_testing -v
```

Allure results are written to `allure-results/`. View them with:

```bash
allure serve allure-results
```

## Notes

- Never commit `.env` or `.env.local` (they can hold API keys). They are in `.gitignore`.
- Where Anjali's and Ankit's versions of the same metric differed, Anjali's version is used
  (for example Answer Relevancy passes at 0.5). The two exceptions are **Prompt Alignment** and
  **Robustness**: Ankit's LLM-judge versions are used by the single-turn tests
  (`prompt_alignment_geval.py`, `robustness_geval.py`), and Anjali's versions
  (`PromptAlignmentMetric`, `SpellingMistakeMetric` / `RobustnessMetric`) stay available for her tests.
