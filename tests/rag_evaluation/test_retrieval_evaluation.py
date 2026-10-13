import allure
import pytest

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

from config.settings import (
    DOCUMENT_PATH,
    VECTOR_DB_PATH,
    RETRIEVAL_DATASET,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    TEMPERATURE,
    THRESHOLD,
)

from utils.data_loader import load_test_data

from rag.document_loader import load_document
from rag.chunker import create_chunks
from rag.vector_store import VectorStore

from rag.retriever import (DocumentRetriever,VectorRetriever)

from agents.research_agent import ResearchAgent
from agents.answer_agent import AnswerAgent
from agents.supervisor_agent import supervisor_agent

from metrics import (
    create_contextual_relevancy_metric,
    create_contextual_precision_metric,
    create_contextual_recall_metric,
)


def test_rag_evaluation():

    # LOAD TEST DATA

    test_data = load_test_data(RETRIEVAL_DATASET)

    # LOAD DOCUMENT

    document = load_document(DOCUMENT_PATH)

    # CREATE CHUNKS

    chunks = create_chunks(document,chunk_size=1)

    # CREATE VECTOR DB
   
    vector_store = VectorStore(VECTOR_DB_PATH)
    vector_store.add_documents(chunks)

    # CREATE RETRIEVERS

    document_retriever = DocumentRetriever(chunks)
    vector_retriever = VectorRetriever(vector_store,threshold=0.60)

    # CREATE OLLAMA MODEL

    evaluation_model = OllamaModel(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE
    )

    # CREATE AGENTS

    research_agent = ResearchAgent(document_retriever,vector_retriever)
    answer_agent = AnswerAgent(evaluation_model)

    # EXECUTE TEST CASES

    for test_case_data in test_data:

        test_case_id = test_case_data["id"]
        question = test_case_data["question"]
        expected_output = test_case_data["expected_output"]
        expected_flow = test_case_data["expected_flow"]

        # READ METRICS FROM DATASET
 
        metric_names = test_case_data.get("metrics", [])

        # ALLURE - TEST DETAILS

        allure.attach(
            question,
            name=f"{test_case_id} - Question",
            attachment_type=allure.attachment_type.TEXT
        )

        allure.attach(
            test_case_data["rag_capability"],
            name=f"{test_case_id} - RAG Capability",
            attachment_type=allure.attachment_type.TEXT
        )

        allure.attach(
            str(metric_names),
            name=f"{test_case_id} - Metrics To Execute",
            attachment_type=allure.attachment_type.TEXT
        )

        # SUPERVISOR AGENT

        result = supervisor_agent(question,research_agent,answer_agent)
        actual_source = result["source"]
        actual_answer = result["answer"]
        retrieval_context = result["retrieval_context"]

        # ATTACH RAG OUTPUT

        allure.attach(
            actual_source,
            name=f"{test_case_id} - Actual Source",
            attachment_type=allure.attachment_type.TEXT
        )

        allure.attach(
            actual_answer,
            name=f"{test_case_id} - Actual Answer",
            attachment_type=allure.attachment_type.TEXT
        )

        allure.attach(
            str(retrieval_context),
            name=f"{test_case_id} - Retrieval Context",
            attachment_type=allure.attachment_type.TEXT
        )

 
        # CREATE LLM TEST CASE

        llm_test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )

        # METRIC RESULTS

        metric_results = {}

        # Metrics that will be passed to DeepEval assert_test()
        executed_metrics = []

        # EXECUTE DATASET-SELECTED METRICS

        if retrieval_context:

            # CONTEXTUAL RELEVANCY
        
            if "contextual_relevancy" in metric_names:

                contextual_relevancy_metric = (create_contextual_relevancy_metric(evaluation_model))
                contextual_relevancy_metric.measure(llm_test_case)
                executed_metrics.append(contextual_relevancy_metric)
                metric_results["contextual_relevancy"] = {
                    "score": contextual_relevancy_metric.score,
                    "reason": contextual_relevancy_metric.reason
                }

            # CONTEXTUAL PRECISION
         
            if "contextual_precision" in metric_names:

                contextual_precision_metric = (create_contextual_precision_metric( evaluation_model))
                contextual_precision_metric.measure(llm_test_case)
                executed_metrics.append(contextual_precision_metric)
                metric_results["contextual_precision"] = {
                    "score": contextual_precision_metric.score,
                    "reason": contextual_precision_metric.reason
                }

            # CONTEXTUAL RECALL

            if "contextual_recall" in metric_names:

                contextual_recall_metric = ( create_contextual_recall_metric(evaluation_model))
                contextual_recall_metric.measure(llm_test_case)
                executed_metrics.append(contextual_recall_metric)
                metric_results["contextual_recall"] = {
                    "score": contextual_recall_metric.score,
                    "reason": contextual_recall_metric.reason
                }

        else:

            # NO RETRIEVAL CONTEXT
         
            for metric_name in metric_names:
                metric_results[metric_name] = {
                    "score": 0,
                    "reason": "Failed - No retrieval context found."
                }

        # PRINT RESULT

        print("\n" + "=" * 70)
        print(f"Test Case ID       : {test_case_id}")
        print(f"RAG Capability     : "f"{test_case_data['rag_capability']}")
        print(f"Metrics To Execute : "f"{metric_names}")
        print(f"Question           : "f"{question}")
        print(f"Expected Flow      : "f"{expected_flow}")
        print(f"Actual Source      : "f"{actual_source}")
        print(f"Actual Answer      : "f"{actual_answer}")
        print(f"Retrieval Context  : "f"{retrieval_context}")

        # PRINT METRIC RESULTS

        print("\nMetrics Result:")
        if metric_results:
            for metric_name, result in metric_results.items():
                print(f"Metric : "f"{metric_name}")
                print(f"Score  : "f"{result['score']}")
                print(f"Reason : "f"{result['reason']}")
                print("-" * 50)
        else:
            print("No metrics configured for this test case.")
            print("=" * 70)

        # ALLURE - METRIC RESULTS

        for metric_name, result in metric_results.items():
            allure.attach(
                f"Metric : {metric_name}\n"
                f"Score  : {result['score']}\n"
                f"Reason : {result['reason']}",
                name=f"{test_case_id} - {metric_name}",
                attachment_type=allure.attachment_type.TEXT
            )

        # VALIDATE SOURCE

        assert actual_source in expected_flow, (
            f"{test_case_id}: "
            f"Actual source '{actual_source}' "
            f"is not present in expected flow "
            f"{expected_flow}"
        )

        # DEEPEVAL ASSERT TEST

        if executed_metrics:
            assert_test(llm_test_case,executed_metrics)

        # CUSTOM THRESHOLD VALIDATION

        for metric_name, result in metric_results.items():
            score = result["score"]
            reason = result["reason"]
            assert score >= THRESHOLD, (
                f"{test_case_id}: "
                f"{metric_name} score {score} < {THRESHOLD}\n"
                f"Reason: {reason}"
            )