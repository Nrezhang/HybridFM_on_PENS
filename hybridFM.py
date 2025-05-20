import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm

# ========== Dataset ==========
class NewsClickDataset(Dataset):
    def __init__(self, interactions, user2idx, item2idx):
        self.user_ids = interactions['user_id'].map(user2idx).values
        self.item_ids = interactions['click_news_id'].map(item2idx).values
        self.labels = np.ones(len(interactions))  # implicit feedback

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user': self.user_ids[idx],
            'item': self.item_ids[idx],
            'label': self.labels[idx]
        }

# ========== Mappings ==========
def create_mappings(train_df, test_df):
    all_users = pd.concat([train_df['user_id'], test_df['user_id']]).unique()
    all_items = pd.concat([train_df['click_news_id'], test_df['click_news_id']]).unique()
    user2idx = {user: idx for idx, user in enumerate(all_users)}
    item2idx = {item: idx for idx, item in enumerate(all_items)}
    return user2idx, item2idx

# ========== FM Models ==========
class FM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(FM, self).__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, item):
        user_vec = self.user_embed(user)
        item_vec = self.item_embed(item)
        dot = torch.sum(user_vec * item_vec, dim=1)
        pred = dot + self.user_bias(user).squeeze() + self.item_bias(item).squeeze() + self.global_bias
        return pred

class HybridFM(nn.Module):
    def __init__(self, num_users, item_embeddings):
        super(HybridFM, self).__init__()
        embedding_dim = item_embeddings.shape[1]
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding.from_pretrained(torch.tensor(item_embeddings, dtype=torch.float32), freeze=True)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(item_embeddings.shape[0], 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, item):
        user_vec = self.user_embed(user)
        item_vec = self.item_embed(item)
        dot = torch.sum(user_vec * item_vec, dim=1)
        pred = dot + self.user_bias(user).squeeze() + self.item_bias(item).squeeze() + self.global_bias
        return pred

# ========== Training ==========
def train_model(model, train_loader, device, epochs=5, lr=0.001):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            user = batch['user'].long().to(device)
            item = batch['item'].long().to(device)
            label = batch['label'].float().to(device)

            preds = model(user, item)
            loss = loss_fn(preds, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss)

# ========== Evaluation ==========
def evaluate_model(model, test_df, user2idx, item2idx, k=10, num_negative=20, device='cpu', max_users=1000, max_pos_per_user=2):
    model.eval()
    model.to(device)
    user_pos_test = test_df.groupby("user_id")["click_news_id"].apply(list).to_dict()
    all_items = list(item2idx.keys())

    sample_users = list(user_pos_test.keys())
    np.random.seed(42)
    sample_users = np.random.choice(sample_users, size=min(max_users, len(sample_users)), replace=False)

    precision_total, recall_total, ndcg_total, mrr_total, user_count = 0, 0, 0, 0, 0

    for user_id in tqdm(sample_users, desc="Evaluating"):
        if user_id not in user2idx:
            continue
        uidx = user2idx[user_id]
        pos_items = user_pos_test[user_id]
        pos_items = np.random.choice(pos_items, size=min(max_pos_per_user, len(pos_items)), replace=False)

        for pos_item in pos_items:
            if pos_item not in item2idx:
                continue
            negatives = np.random.choice([i for i in all_items if i not in pos_items], num_negative, replace=False)
            candidates = [pos_item] + list(negatives)
            labels = [1] + [0] * num_negative
            iidx = [item2idx[i] for i in candidates]

            user_tensor = torch.tensor([uidx] * len(candidates)).to(device)
            item_tensor = torch.tensor(iidx).to(device)

            with torch.no_grad():
                scores = model(user_tensor, item_tensor).cpu().numpy()

            ranked_indices = np.argsort(scores)[::-1]
            ranked_labels = np.array(labels)[ranked_indices]

            # Metrics
            top_k = ranked_labels[:k]
            precision = np.sum(top_k) / k
            recall = np.sum(top_k)
            dcg = np.sum(top_k / np.log2(np.arange(2, 2 + k)))
            ndcg = dcg  # IDCG = 1 for a single positive

            try:
                rank = np.where(ranked_labels == 1)[0][0] + 1
                mrr = 1 / rank
            except IndexError:
                mrr = 0

            precision_total += precision
            recall_total += recall
            ndcg_total += ndcg
            mrr_total += mrr
            user_count += 1

    print(f"\nEvaluation Results on {user_count} users (sampled):")
    print(f"  Precision@{k}: {precision_total / user_count:.4f}")
    print(f"  Recall@{k}:    {recall_total / user_count:.4f}")
    print(f"  NDCG@{k}:      {ndcg_total / user_count:.4f}")
    print(f"  MRR:           {mrr_total / user_count:.4f}")



# ========== Main ==========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    train_df = pd.read_pickle("train_split.pkl")
    test_df = pd.read_pickle("test_split.pkl")
    with open("news_embeddings.pkl", "rb") as f:
        news_embeds = pickle.load(f)

    user2idx, item2idx = create_mappings(train_df, test_df)

    print("Preparing hybrid embeddings...")
    embed_dim = len(next(iter(news_embeds.values())))
    item_embedding_matrix = np.zeros((len(item2idx), embed_dim), dtype=np.float32)
    for news_id, idx in item2idx.items():
        if news_id in news_embeds:
            item_embedding_matrix[idx] = news_embeds[news_id]
        else:
            item_embedding_matrix[idx] = np.random.normal(0, 0.1, embed_dim).astype(np.float32)

    train_dataset = NewsClickDataset(train_df, user2idx, item2idx)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # Train Standard FM
    print("\nTraining Standard FM...")
    fm_model = FM(num_users=len(user2idx), num_items=len(item2idx), embedding_dim=32)
    # train_model(fm_model, train_loader, device=device, epochs=5, lr=0.001)
    # torch.save(fm_model.state_dict(), "standard_fm.pt")

    # Train Hybrid FM
    print("\nTraining Hybrid FM...")
    hybrid_model = HybridFM(num_users=len(user2idx), item_embeddings=item_embedding_matrix)
    # train_model(hybrid_model, train_loader, device=device, epochs=5, lr=0.001)
    # torch.save(hybrid_model.state_dict(), "hybrid_fm.pt")

    # Evaluation
    print("\nEvaluating Standard FM:")
    fm_model.load_state_dict(torch.load("standard_fm.pt"))
    evaluate_model(fm_model, test_df, user2idx, item2idx, device=device)

    print("\nEvaluating Hybrid FM:")
    hybrid_model.load_state_dict(torch.load("hybrid_fm.pt"))
    evaluate_model(hybrid_model, test_df, user2idx, item2idx, device=device)

if __name__ == "__main__":
    main()
