import pytest


if __name__ == "__main__":
    exit_code = pytest.main(
        [
            "tests",
            "-v",
            "-s"
        ]
    )

    raise SystemExit(exit_code)