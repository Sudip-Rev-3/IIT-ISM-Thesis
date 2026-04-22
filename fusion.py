import torch.nn.functional as F
import torch


def _align_sequence_tensors(inputs):
    if "input_ids" not in inputs:
        return inputs

    target_len = inputs["input_ids"].shape[1]
    for key, value in list(inputs.items()):
        if not torch.is_tensor(value):
            continue
        if value.ndim < 2:
            continue
        if value.shape[1] == target_len:
            continue
        inputs[key] = value[:, :target_len]

    return inputs

def _normalize_weights(lambda_v, lambda_c):
    total_lambda = max(lambda_v + lambda_c, 1e-6)
    visual_weight = max(lambda_v, 0.0) / total_lambda
    context_weight = max(lambda_c, 0.0) / total_lambda
    return visual_weight, context_weight


def fuse_logits(logits_p, logits_c, lambda_v, lambda_c):
    fused = []
    visual_weight, context_weight = _normalize_weights(lambda_v, lambda_c)

    for lp, lc in zip(logits_p, logits_c):
        lp = lp[0]
        lc = lc[0]

        fused_logit = (visual_weight * lp) + (context_weight * lc)
        fused.append(fused_logit.unsqueeze(0))
    return fused


def decode_from_logits(fused_logits, processor):
    tokens = []

    for logit in fused_logits:
        probs = F.softmax(logit, dim=-1)
        token = torch.argmax(probs, dim=-1)
        tokens.append(token)

    tokens = torch.cat(tokens, dim=0)

    return processor.decode(tokens, skip_special_tokens=True)


def _prepare_step_inputs(base_inputs, generated_ids):
    step_inputs = {k: v for k, v in base_inputs.items()}

    prompt_input_ids = base_inputs["input_ids"]
    prompt_attn = base_inputs.get("attention_mask")
    if prompt_attn is None:
        prompt_attn = torch.ones_like(prompt_input_ids)

    if generated_ids.numel() == 0:
        step_inputs["input_ids"] = prompt_input_ids
        step_inputs["attention_mask"] = prompt_attn
        return step_inputs

    generated_attn = torch.ones_like(generated_ids)
    step_inputs["input_ids"] = torch.cat([prompt_input_ids, generated_ids], dim=1)
    step_inputs["attention_mask"] = torch.cat([prompt_attn, generated_attn], dim=1)
    return step_inputs


def decode_with_stepwise_fusion(
    model,
    processor,
    image_path,
    question,
    context,
    lambda_v,
    lambda_c,
    max_new_tokens=200,
):
    visual_weight, context_weight = _normalize_weights(lambda_v, lambda_c)

    visual_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    context_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{context}. So {question}"},
            ],
        }
    ]

    visual_prompt = processor.apply_chat_template(visual_messages, add_generation_prompt=True)
    context_prompt = processor.apply_chat_template(context_messages, add_generation_prompt=True)

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    first_param = next(model.parameters())
    device = first_param.device
    dtype = first_param.dtype

    visual_inputs = processor(images=[image], text=[visual_prompt], padding=True, return_tensors="pt")
    context_inputs = processor(images=[image], text=[context_prompt], padding=True, return_tensors="pt")

    visual_inputs = _align_sequence_tensors(visual_inputs)
    context_inputs = _align_sequence_tensors(context_inputs)

    for key in visual_inputs:
        visual_inputs[key] = visual_inputs[key].to(device)
    for key in context_inputs:
        context_inputs[key] = context_inputs[key].to(device)

    if "pixel_values" in visual_inputs:
        visual_inputs["pixel_values"] = visual_inputs["pixel_values"].to(dtype=dtype)
    if "pixel_values" in context_inputs:
        context_inputs["pixel_values"] = context_inputs["pixel_values"].to(dtype=dtype)

    generated_ids = torch.empty((1, 0), dtype=visual_inputs["input_ids"].dtype, device=device)
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            v_step_inputs = _prepare_step_inputs(visual_inputs, generated_ids)
            c_step_inputs = _prepare_step_inputs(context_inputs, generated_ids)

            visual_outputs = model(**v_step_inputs, return_dict=True)
            context_outputs = model(**c_step_inputs, return_dict=True)

            visual_next = visual_outputs.logits[:, -1, :]
            context_next = context_outputs.logits[:, -1, :]
            fused_next = (visual_weight * visual_next) + (context_weight * context_next)

            next_token = torch.argmax(fused_next, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

    if generated_ids.numel() == 0:
        return ""

    return processor.decode(generated_ids[0], skip_special_tokens=True)
