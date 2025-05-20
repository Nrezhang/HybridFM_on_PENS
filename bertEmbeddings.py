from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from ingest import load_data, temporal_split, print_info
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os
import pickle

train, valid, news, test = load_data()
train_split, test_split = temporal_split(train)

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = BertModel.from_pretrained('bert-base-uncased').to(device)
model.eval()

def get_bert_embedding(text):
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return None

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze().cpu().numpy()  # move back to CPU for numpy


# news_df: must contain 'news_id' and 'text' (title/content)
def create_embeddings(news_df):
    # Ensure the DataFrame contains the required columns
    if 'news_id' not in news_df.columns or 'full_text' not in news_df.columns:
        raise ValueError("DataFrame must contain 'news_id' and 'text' columns.")

    # Create embeddings
    embeddings = {}
    for _, row in news_df.iterrows():
        news_id = row['news_id']
        text = row['full_text']  # or title + abstract
        embeddings[news_id] = get_bert_embedding(text)
    return embeddings
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def recommend_top_k(user_embedding, news_embeddings, k=10):
    news_ids = list(news_embeddings.keys())
    embeddings = np.stack([news_embeddings[nid] for nid in news_ids])

    sims = cosine_similarity([user_embedding], embeddings)[0]
    top_k_idx = sims.argsort()[-k:][::-1]
    return [news_ids[i] for i in top_k_idx]
    # news_df: must contain 'news_id' and 'text' (title/content)

def safe_load_pickle(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        print(f"Warning: Failed to load {path}. Starting fresh.")
        return {}

def generate_news_embeddings(news_df, embed_fn, batch_size=128, use_cache=True, save_path="news_embeddings.pkl"):
    # Load existing embeddings or initialize fresh
    news_embeddings = safe_load_pickle(save_path) if use_cache else {}
    processed_ids = set(news_embeddings.keys())

    for i in tqdm(range(0, len(news_df), batch_size), desc="Processing Batches"):
        batch = news_df.iloc[i:i + batch_size]
        batch_embeddings = {}

        for _, row in (batch.iterrows()):
            news_id = row['news_id']
            if news_id in processed_ids:
                continue

            text = row['full_text']
            embedding = embed_fn(text)
            if embedding is not None:
                batch_embeddings[news_id] = embedding

        # Save updated embeddings to file after every batch
        news_embeddings.update(batch_embeddings)
        with open(save_path, "wb") as f:
            pickle.dump(news_embeddings, f)
    return news_embeddings

if __name__ == "__main__":

    news_embeddings = generate_news_embeddings(news, get_bert_embedding)


    user_profiles = defaultdict(list)

    for _, row in train_split.iterrows():
        user_id = row['user_id']
        news_id = row['click_news_id']
        if news_id in news_embeddings:
            user_profiles[user_id].append(news_embeddings[news_id])
    for user_id in user_profiles:
        user_profiles[user_id] = np.mean(user_profiles[user_id], axis=0) #sum embeddings for one user

    user_id = 'N27499'
    user_embed = user_profiles[user_id]
    top_k_items = recommend_top_k(user_embed, news_embeddings, k=10)

