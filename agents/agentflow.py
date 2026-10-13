import json

import ollama

from config.settings import ANSWER_MODEL_NAME


def search_iphone_catalog(query):
    return (
        "Model: iPhone 16. Available colors: Ultramarine, Teal, Pink, "
        "White, and Black. Price: starts at $799 USD."
    )


def search_samsung_catalog(query):
    return (
        "Model: Samsung Galaxy S25. Available colors: Navy, Silver Shadow, "
        "Icyblue, and Mint. Price: starts at $799.99 USD."
    )


def run_tool(tool_name, tool_args):
    if tool_name == "search_iphone_catalog":
        return search_iphone_catalog(tool_args["query"])

    if tool_name == "search_samsung_catalog":
        return search_samsung_catalog(tool_args["query"])

    return "Unknown tool"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_iphone_catalog",
            "description": "Search the e-commerce product catalog for iPhone product details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "iPhone product search query, such as iPhone 16."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_samsung_catalog",
            "description": "Search the e-commerce product catalog for Samsung phone details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Samsung phone product search query, such as Samsung Galaxy S25."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def run_agentflow(prompt, max_steps=5, response_callback=None):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    tools_used = []
    trace = []

    for _ in range(max_steps):

        response = ollama.chat(
            model=ANSWER_MODEL_NAME,
            messages=messages,
            tools=TOOLS
        )

        message = response["message"]
        print(
            "========================================\n"
            "AGENT MODEL RESPONSE:\n"
            f"{json.dumps(message, ensure_ascii=False, default=str)}\n"
            "========================================",
            flush=True,
        )

        if response_callback is not None:
            response_callback(message)

        if message.get("tool_calls"):

            for call in message["tool_calls"]:

                tool_name = call["function"]["name"]
                tool_args = call["function"]["arguments"]

                print(
                    "\n========================================\n"
                    f"EXECUTING TOOL: {tool_name}\n"
                    f"TOOL ARGUMENTS: {json.dumps(tool_args, ensure_ascii=False)}\n"
                    "========================================\n",
                    flush=True,
                )

                tool_result = run_tool(
                    tool_name,
                    tool_args
                )

                tools_used.append(tool_name)

                trace.append(
                    {
                        "name": tool_name,
                        "input_parameters": tool_args,
                        "output": tool_result
                    }
                )

                messages.append(message)

                messages.append(
                    {
                        "role": "tool",
                        "content": tool_result
                    }
                )

        else:

            return (
                message.get("content", ""),
                tools_used,
                trace
            )

    return (
        "Stopped after max steps without finishing.",
        tools_used,
        trace
    )


def generate_plan(prompt):

    return (
        "1. Search the Playwright coding guidelines.\n"
        "2. Identify the recommended locator strategy.\n"
        "3. Generate the Playwright script using that guideline."
    )


def run_llm_agentflow(prompt):

    answer, tools_used, trace = run_agentflow(prompt)

    return answer, tools_used, trace
