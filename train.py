import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from acars.data_loader import YelpDataset
from acars.model import ACARSModel
from acars.contrastive_loss import contrastive_loss

# Set file paths for Yelp JSON (update paths if needed)
REVIEW_PATH = "datasets/yelp_academic_dataset_review.json"
USER_PATH = "datasets/yelp_academic_dataset_user.json"
BUSINESS_PATH = "datasets/yelp_academic_dataset_business.json"

# Load dataset
dataset = YelpDataset(REVIEW_PATH, USER_PATH, BUSINESS_PATH)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ACARSModel(num_users=len(dataset.user_enc.classes_),
                   num_items=len(dataset.item_enc.classes_)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

# 🔁 TRAINING LOOP
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        user = batch['user'].to(device)
        item = batch['item'].to(device)
        timestamp = batch['timestamp'].to(device)
        rating_true = batch['rating'].to(device)

        # Forward pass
        rating_pred, ctr_pred, engage_pred, contrastive_vec = model(user, item, timestamp)

        # Fake contrastive pair (real use would involve augmentations)
        cl_loss = contrastive_loss(contrastive_vec, contrastive_vec)

        # Multi-task losses
        loss_rating = mse_loss(rating_pred, rating_true)
        loss_ctr = bce_loss(ctr_pred, (rating_true >= 4).float())
        loss_engage = mse_loss(engage_pred, rating_true)  # using rating as proxy for engagement

        # Combine all
        loss = loss_rating + loss_ctr + loss_engage + 0.1 * cl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")
