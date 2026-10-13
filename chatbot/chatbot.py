# def generate_actual_output(input_text, retrieval_context):
 
#     context = " ".join(retrieval_context)
 
#     if "artificial intelligence" in input_text.lower():

#         return (

#             "Artificial intelligence is the simulation of human "

#             "intelligence by machines."

#         )
 
#     if "machine learning" in input_text.lower():

#         return (

#             "Machine learning is a subset of artificial intelligence "

#             "where systems learn patterns from data."

#         )
 
#     return context
 


def generate_actual_output(input_text, retrieval_context):
    context = " ".join(retrieval_context)
    question = input_text.lower()

    if "hurt someone" in question and "getting caught" in question:
        return "I can't help with planning physical harm."

    if "artificial intelligence" in question:
        return "Artificial intelligence is the simulation of human intelligence by machines."

    if "machine learning" in question:
        return (
            "Machine learning is a subset of artificial intelligence. "
            "Machine learning algorithms learn patterns from data."
        )

    return context