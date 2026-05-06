from load_llava import load_model
from realloc_attn import calculate_similarity
from inference import generate_response


def _choose_final_answer(fused_answer, visual_answer, context_answer, lambda_table, lambda_context):
    if lambda_table > lambda_context and visual_answer.strip():
        return visual_answer

    if fused_answer.strip():
        return fused_answer

    if context_answer.strip():
        return context_answer

    if visual_answer.strip():
        return visual_answer

    return context_answer


def attn_realloc_ans(
    image,
    question,
    context,
    visual_priority=3.0,
    model=None,
    processor=None,
    similarity_model=None,
    visual_answer=None,
    visual_logits=None,
):
    if model is None or processor is None:
        model, processor = load_model()

    # Step 1: Base visual-only answer.
    if visual_answer is None or visual_logits is None:
        response_without_context, _, _ = generate_response(model, processor, image, question)
    else:
        response_without_context = visual_answer

    # Step 2: Similarity-based visual weight.
    context_score, table_score, lambda_context, lambda_table = calculate_similarity(
        question,
        context,
        image,
        visual_priority=visual_priority,
        model=similarity_model,
    )

    # Step 3: Reallocate attention by boosting visual token weights.
    visual_token_boost = 1.0 + (lambda_table * visual_priority)
    final_answer, _, _ = generate_response(
        model,
        processor,
        image,
        question,
        context,
        visual_token_boost=visual_token_boost,
    )

    final_answer = _choose_final_answer(
        final_answer,
        response_without_context,
        "",
        lambda_table,
        lambda_context,
    )
    return final_answer


if __name__ == "__main__":
    image_path = "images/test8.jpg"
    question = "Which country has the highest percentage for watching the World Cup?"
    context = "Germany leads in viewer interest across all categories. The national enthusiasm for the World Cup is reflected in these consistently high survey numbers. No other country reaches the same level of engagement in this study."
    final_answer = attn_realloc_ans(image_path, question, context)
    print(f"Final Answer: {final_answer}")
