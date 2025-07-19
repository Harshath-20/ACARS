import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score
import numpy as np

from acars.data_loader import YelpDataset
from acars.model import ACARSModel

# Load data
REVIEW_PATH = "datasets/yelp_academic_dataset_review.json"
USER_PATH = "datasets/yelp_academic_dataset_user.json"
BUSINESS_PATH = "datasets/yelp_academic_dataset_business.json"

dataset = YelpDataset(REVIEW_PATH, USER_PATH, BUSINESS_PATH)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ACARSModel(num_users=len(dataset.user_enc.classes_),
                   num_items=len(dataset.item_enc.classes_)).to(device)
model.load_state_dict(torch.load("acars_model.pth"))  # Load your trained model here
model.eval()

ratings_true = []
ratings_pred = []
ctrs_true = []
ctrs_pred = []

with torch.no_grad():
    for batch in dataloader:
        user = batch['user'].to(device)
        item = batch['item'].to(device)
        timestamp = batch['timestamp'].to(device)
        rating_true = batch['rating'].cpu().numpy()

        rating_pred, ctr_pred, _, _ = model(user, item, timestamp)
        ratings_pred.extend(rating_pred.cpu().numpy())
        ratings_true.extend(rating_true)
        ctrs_pred.extend(ctr_pred.cpu().numpy())
        ctrs_true.extend((rating_true >= 4).astype(int))  # Binary relevance

# Metrics
rmse = np.sqrt(mean_squared_error(ratings_true, ratings_pred))
precision = precision_score(ctrs_true, np.array(ctrs_pred) > 0.5)
auc = roc_auc_score(ctrs_true, ctrs_pred)

print(f"RMSE: {rmse:.4f}")
print(f"Precision@1: {precision:.4f}")
print(f"AUC (CTR): {auc:.4f}")
