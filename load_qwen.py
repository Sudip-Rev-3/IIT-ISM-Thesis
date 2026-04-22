from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from dotenv import load_dotenv
load_dotenv()


def load_model(model_id: str = "Qwen/Qwen3.5-0.8B", device: str = "cpu"):
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    target_device = "cuda" if use_cuda else "cpu"

    common_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "device_map": "auto" if use_cuda else "cpu",
    }

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **common_kwargs)
    except RuntimeError as exc:
        # Some checkpoints can fail strict shape checks across transformers versions.
        if "ignore_mismatched_sizes" not in str(exc):
            raise
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            **common_kwargs,
        )

    if not use_cuda:
        model = model.to(target_device)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor