import pytest


@pytest.fixture(scope="session", autouse=True)
def framework_setup():
    """
    Framework-level setup.
    """

    print("\n")
    print("=" * 60)
    print("Starting LLM Evaluation Framework")
    print("=" * 60)

    yield

    print("\n")
    print("=" * 60)
    print("LLM Evaluation Framework Execution Completed")
    print("=" * 60)