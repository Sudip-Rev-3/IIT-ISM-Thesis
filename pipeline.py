from load_qwen import load_model
from fusion import fuse_logits, decode_from_logits, decode_with_stepwise_fusion
from realloc_attn import calculate_similarity
from inference import generate_response


def _choose_final_answer(fused_answer, visual_answer, context_answer):
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
    fusion_mode="phase1",
):
    if model is None or processor is None:
        model, processor = load_model()

    # Step 1: Generate response and get transition scores
    if visual_answer is None or visual_logits is None:
        response_without_context, avg_score_without_context, logits_p = generate_response(model, processor, image, question)
    else:
        response_without_context = visual_answer
        logits_p = visual_logits

    response_with_context, avg_score_with_context, logits_c = generate_response(model, processor, image, question, context)
    # Step 2: Calculate similarity scores for attention reallocation
    context_score, table_score, lambda_context, lambda_table = calculate_similarity(
        question,
        context,
        image,
        visual_priority=visual_priority,
        model=similarity_model,
    )

    # Step 3: Fuse and decode according to selected phase.
    if fusion_mode == "phase2":
        final_answer = decode_with_stepwise_fusion(
            model,
            processor,
            image,
            question,
            context,
            lambda_v=lambda_table,
            lambda_c=lambda_context,
        )
    else:
        fused_logits = fuse_logits(logits_p, logits_c, lambda_v=lambda_table, lambda_c=lambda_context)
        final_answer = decode_from_logits(fused_logits, processor)

    final_answer = _choose_final_answer(
        final_answer,
        response_without_context,
        response_with_context,
    )
    return final_answer


if __name__ == "__main__":
    image_path = "images/test3.jpg"
    question = "On which day were the push-ups the lowest?"
    context = "The lowest performance happens on Monday due to fatigue."
    final_answer = attn_realloc_ans(image_path, question, context)
    print(f"Final Answer: {final_answer}")
