from deepeval.metrics import PromptAlignmentMetric
 
 
def get_prompt_alignment_metric(model, prompt_instructions):
    return PromptAlignmentMetric(
        prompt_instructions=prompt_instructions,
        threshold=0.0,
        model=model,
        include_reason=True
    )