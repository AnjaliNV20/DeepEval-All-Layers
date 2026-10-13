"""One-off check that the local Ollama model can return DeepEval's Claims schema.

Run it by hand:  python scripts/check_faithfulness_schema.py
(It is not a pytest test, so it is not run with the test suite.)
"""

import asyncio

from deepeval.models import OllamaModel
from deepeval.metrics.faithfulness.schema import Claims


model = OllamaModel(
    model="llama3.1:latest",
    base_url="http://localhost:11434",
)


async def main():
    result = await model.a_generate(
        "Extract one claim from this text: Playwright is a browser automation tool.",
        schema=Claims,
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
