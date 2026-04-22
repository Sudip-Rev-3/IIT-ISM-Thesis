import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer, util

from inference import generate_response
from load_qwen import load_model


DEFAULT_OUTPUT = "qwen_answers.csv"


def load_examples(data_path):
    with open(data_path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_similarity(clip_model, question, context, image_path, visual_priority):
    q_emb = clip_model.encode(question)
    context_emb = clip_model.encode(context)

    image = Image.open(image_path).convert("RGB")
    table_emb = clip_model.encode(image)

    context_score = util.cos_sim(q_emb, context_emb).item()
    table_score = util.cos_sim(q_emb, table_emb).item()

    context_weight = max(context_score, 0.0)
    table_weight = max(table_score, 0.0) * visual_priority
    total_weight = context_weight + table_weight

    if total_weight == 0:
        return 0.5, 0.5

    lambda_context = context_weight / total_weight
    lambda_table = table_weight / total_weight
    return lambda_context, lambda_table


def choose_final_answer(visual_answer, context_answer, lambda_table, lambda_context):
    if lambda_table >= lambda_context and visual_answer.strip():
        return visual_answer

    if context_answer.strip():
        return context_answer

    return visual_answer


def generate_realloc_attn_answer(
    model,
    processor,
    clip_model,
    image_path,
    question,
    context,
    visual_priority,
    visual_answer=None,
):
    if visual_answer is None:
        visual_answer, _, _ = generate_response(model, processor, image_path, question)

    lambda_context, lambda_table = calculate_similarity(
        clip_model,
        question,
        context,
        image_path,
        visual_priority,
    )

    visual_token_boost = 1.0 + (lambda_table * visual_priority)
    realloc_answer, _, _ = generate_response(
        model,
        processor,
        image_path,
        question,
        context,
        visual_token_boost=visual_token_boost,
    )

    if realloc_answer.strip():
        return realloc_answer

    return choose_final_answer(
        visual_answer,
        "",
        lambda_table,
        lambda_context,
    )


def build_results(data_path, images_dir, visual_priority, limit=None, device="cpu"):
    examples = load_examples(data_path)
    if limit is not None:
        examples = examples[:limit]

    model, processor = load_model(device=device)
    clip_model = SentenceTransformer("clip-ViT-B-32")
    rows = []

    for index, example in enumerate(examples, start=1):
        image_path = images_dir / example["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for id {example['id']}: {image_path}")

        print(f"[{index}/{len(examples)}] id={example['id']} image={example['image']}")

        no_fusion_answer, _, _ = generate_response(
            model,
            processor,
            str(image_path),
            example["question"],
        )
        realloc_attn_answer = generate_realloc_attn_answer(
            model,
            processor,
            clip_model,
            str(image_path),
            example["question"],
            example["context"],
            visual_priority,
            visual_answer=no_fusion_answer,
        )

        rows.append(
            {
                "id": example["id"],
                "question": example["question"],
                "actual ans": example["answer"],
                "model ans base": no_fusion_answer,
                "model ans with realloc attn": realloc_attn_answer,
            }
        )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate base vs attention-reallocation answer CSV from data.json."
    )
    parser.add_argument("--data", default="data.json", help="Path to the JSON dataset.")
    parser.add_argument("--images-dir", default="images", help="Directory containing images.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument(
        "--visual-priority",
        type=float,
        default=3.0,
        help="Multiplier for image/table similarity during attention reallocation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process for a quick smoke test.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Execution device. Use cuda on Colab T4; cpu for local runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataframe = build_results(
        data_path=Path(args.data),
        images_dir=Path(args.images_dir),
        visual_priority=args.visual_priority,
        limit=args.limit,
        device=args.device,
    )
    dataframe.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(dataframe)} rows to {args.output}")
