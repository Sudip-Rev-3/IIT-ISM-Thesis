from load_qwen import load_model
from fusion import fuse_logits, decode_from_logits

from PIL import Image
import torch

def generate_response(model, processor, image_path, question, context=None):
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