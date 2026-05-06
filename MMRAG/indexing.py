import faiss
from embed import extract_embeddings

def norm_embed(embeddings):
    faiss.normalize_L2(embeddings)
    return embeddings

def build_index(image_embeddings, text_embeddings):
    # Normalize embeddings
    image_embeddings = norm_embed(image_embeddings)
    text_embeddings = norm_embed(text_embeddings)

    # Build FAISS index for image embeddings
    image_index = faiss.IndexFlatL2(image_embeddings.shape[1])
    image_index.add(image_embeddings)

    # Build FAISS index for text embeddings
    text_index = faiss.IndexFlatL2(text_embeddings.shape[1])
    text_index.add(text_embeddings)

    return image_index, text_index