from load_qwen import load_model

from PIL import Image
import torch


def _collect_visual_token_ids(processor):
    token_ids = set()
    tokenizer = getattr(processor, "tokenizer", None)

    for attr in ["image_token_id", "img_token_id", "vision_token_id"]:
        value = getattr(processor, attr, None)
        if isinstance(value, int):
            token_ids.add(value)

    if tokenizer is not None:
        candidate_tokens = [
            "<image>",
            "<img>",
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ]
        for token in candidate_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                token_ids.add(token_id)

        added_vocab = tokenizer.get_added_vocab()
        for token, token_id in added_vocab.items():
            lowered = token.lower()
            if "image" in lowered or "vision" in lowered or "img" in lowered:
                token_ids.add(token_id)

    return token_ids


def _boost_visual_attention(inputs, processor, visual_token_boost):
    if visual_token_boost <= 1.0:
        return inputs

    input_ids = inputs.get("input_ids")
    attn_mask = inputs.get("attention_mask")
    if input_ids is None or attn_mask is None:
        return inputs

    visual_token_ids = _collect_visual_token_ids(processor)
    if not visual_token_ids:
        return inputs

    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in visual_token_ids:
        visual_mask |= input_ids == token_id

    if not visual_mask.any():
        return inputs

    boosted_attn = attn_mask.to(torch.float32).clone()
    boosted_attn[visual_mask] = boosted_attn[visual_mask] * float(visual_token_boost)
    inputs["attention_mask"] = boosted_attn
    return inputs


def generate_response(
    model,
    processor,
    image_path,
    question,
    context=None,
    visual_token_boost=1.0,
):
    if context:
        prompt = f"{context}. So {question}"
    else:
        prompt = question

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cpu", torch.float32)
    inputs = _boost_visual_attention(inputs, processor, visual_token_boost)

    output = model.generate(
        **inputs, 
        max_new_tokens=200, 
        do_sample=False, 
        return_dict_in_generate=True,
        output_scores=True
    )

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = output.sequences[:, prompt_length:]
    response = processor.decode(generated_tokens[0], skip_special_tokens=True)
    transition_scores = model.compute_transition_scores(
        output.sequences,
        output.scores,
        normalize_logits=True
    )
    avg_score = transition_scores.mean().item()
    return response, avg_score, output.scores

if __name__ == "__main__":
    image_path = "images/test3.jpg"
    question = "On which day were the push-ups the lowest?"
    context = "The lowest performance happens on Monday due to fatigue."
    model, processor = load_model()
    response, avg_score, logits = generate_response(model, processor, image_path, question)
    print("Response:", response)
