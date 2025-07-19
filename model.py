import torch
import torch.nn as nn
import torch.nn.functional as F

class ACARSModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(ACARSModel, self).__init__()

        # Embedding layers for user, item, and time
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.time_proj = nn.Linear(1, embedding_dim)

        # Transformer encoder to process combined input
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4),
            num_layers=2
        )

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Multi-task heads
        self.rating_head = nn.Linear(embedding_dim, 1)       # Regression
        self.ctr_head = nn.Linear(embedding_dim, 1)          # Binary (click-through rate)
        self.engagement_head = nn.Linear(embedding_dim, 1)   # Proxy for engagement (time/session)

    def forward(self, user, item, timestamp):
        # Embeddings
        u = self.user_emb(user)
        i = self.item_emb(item)
        t = self.time_proj(timestamp.unsqueeze(1))  # Project time as feature

        # Combine into a sequence and feed to transformer
        x = torch.stack([u, i, t], dim=1)           # Shape: (batch, seq=3, dim)
        rep = self.transformer(x).mean(dim=1)       # Mean pooling

        # Contrastive representation
        contrastive_vec = self.projector(rep)

        # Multi-task outputs
        rating = self.rating_head(rep).squeeze(1)
        ctr = torch.sigmoid(self.ctr_head(rep)).squeeze(1)
        engagement = self.engagement_head(rep).squeeze(1)

        return rating, ctr, engagement, contrastive_vec
