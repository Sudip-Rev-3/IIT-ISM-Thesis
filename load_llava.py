import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from dotenv import load_dotenv
load_dotenv()

def load_model(model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device: str = "cpu"):
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to("cuda" if use_cuda else "cpu")

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

if __name__ == "__main__":
    model, processor = load_model()