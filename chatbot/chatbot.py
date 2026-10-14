from config.models import get_answer_model

def generate_ui_output(input_text, prompt_instructions):
    instructions = "\n".join(f"- {instruction}" for instruction in prompt_instructions)
    prompt = (
        "You are a helpful assistant. Follow the response instructions below.\n\n"
        f"Response instructions:\n{instructions}\n\n"
        f"User question:\n{input_text}\n\n"
        "Answer:"
    )

    response, _ = get_answer_model().generate(prompt)
    return str(response)