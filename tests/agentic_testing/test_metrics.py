from .runners.tool_correctness import run_tool_correctness
from .runners.task_completion import run_task_completion
from .runners.argument_correctness import run_argument_correctness
from .runners.plan_adherence import run_plan_adherence
from .runners.plan_quality import run_plan_quality


def test_tool_correctness():

    score = run_tool_correctness()

    print("\nTool Correctness Score:", score)

    assert score >= 0.5


def test_task_completion():

    score = run_task_completion()

    print("\nTask Completion Score:", score)

    assert score >= 0.5


def test_argument_correctness():

    score = run_argument_correctness()

    print("\nArgument Correctness Score:", score)

    assert score >= 0.5


def test_plan_adherence():

    score = run_plan_adherence()

    print("\nPlan Adherence Score:", score)

    assert score >= 0.5


def test_plan_quality():

    score = run_plan_quality()

    print("\nPlan Quality Score:", score)

    assert score >= 0.5