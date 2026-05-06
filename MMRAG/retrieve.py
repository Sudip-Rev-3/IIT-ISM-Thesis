import torch
import faiss
from load_clip import model, processor, device
from embed import extract_embeddings
from indexing import build_index

def retrieve(query, data, alpha=0.7, beta=0.3, top_k=5):
    
    # Encode query (TEXT ONLY)
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        query_emb = model.get_text_features(**inputs)
    
    query_emb = query_emb.cpu().numpy().astype("float32")
    faiss.normalize_L2(query_emb)
    
    image_embeddings, text_embeddings = extract_embeddings(data)
    image_index, text_index = build_index(image_embeddings, text_embeddings)
    # Search both indexes
    D_img, I_img = image_index.search(query_emb, top_k)
    D_txt, I_txt = text_index.search(query_emb, top_k)
    
    # Combine scores
    scores = {}
    
    for i in range(top_k):
        idx_img = I_img[0][i]
        idx_txt = I_txt[0][i]
        
        scores[idx_img] = scores.get(idx_img, 0) + alpha * D_img[0][i]
        scores[idx_txt] = scores.get(idx_txt, 0) + beta * D_txt[0][i]
    
    # Sort
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    results = [data[idx] for idx, _ in ranked[:top_k]]
    
    return results