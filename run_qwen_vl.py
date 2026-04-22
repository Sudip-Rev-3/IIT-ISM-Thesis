from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", 
    dtype=torch.float32, 
    device_map="cpu"
)

image = Image.open("images/test1.jpg")

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

question = "What is the profit or loss at 300 units of output?"
context = "At 300 units, the company already earns a profit of $3,000."
prompt = f"{context}. So {question}"
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

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)