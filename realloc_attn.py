from functools import lru_cache

from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from PIL import Image
load_dotenv()

CLIP_MODEL_NAME = "clip-ViT-B-32"


@lru_cache(maxsize=1)
def load_similarity_model():
    return SentenceTransformer(CLIP_MODEL_NAME)


def calculate_similarity(question, context, raw_image, visual_priority=3.0, model=None):
    if model is None:
        model = load_similarity_model()

    q_emb = model.encode(question)
    context_emb = model.encode(context)
    image = Image.open(raw_image).convert("RGB")
    table_emb = model.encode(image)

    context_score = util.cos_sim(q_emb, context_emb)
    table_score = util.cos_sim(q_emb, table_emb)

    context_weight = max(context_score.item(), 0.0)
    table_weight = max(table_score.item(), 0.0) * visual_priority
    total_weight = context_weight + table_weight

    if total_weight == 0:
        lambda_context = 0.5
        lambda_table = 0.5
    else:
        lambda_context = context_weight / total_weight
        lambda_table = table_weight / total_weight
    print(f"Context Score: {context_score.item():.4f}, Table Score: {table_score.item():.4f}")
    print(f"Lambda Context: {lambda_context:.4f}, Lambda Table: {lambda_table:.4f}")

    return context_score, table_score, lambda_context, lambda_table


if __name__ == "__main__":
    question = "Which day has total 10 push ups ?"
    context = "Friday has zero push ups but saturday has 10 push ups."
    raw_image = "images/test3.jpg"
    context_score, table_score, lambda_context, lambda_table = calculate_similarity(question, context, raw_image)
