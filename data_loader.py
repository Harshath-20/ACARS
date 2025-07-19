import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

class YelpDataset(Dataset):
    def __init__(self, review_path, user_path, business_path):
        # Load JSON files
        self.reviews = self._load_json(review_path)
        self.users = self._load_json(user_path)
        self.businesses = self._load_json(business_path)

        # Convert reviews to DataFrame
        self.df = pd.DataFrame(self.reviews)
        self.df = self.df[['user_id', 'business_id', 'stars', 'date']]
        self.df = self.df.dropna()

        # Label encode user and item IDs
        self.user_enc = LabelEncoder().fit(self.df['user_id'])
        self.item_enc = LabelEncoder().fit(self.df['business_id'])

        self.df['user'] = self.user_enc.transform(self.df['user_id'])
        self.df['item'] = self.item_enc.transform(self.df['business_id'])
        self.df['timestamp'] = pd.to_datetime(self.df['date']).astype(int) // 10**9  # UNIX timestamp

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'user': torch.tensor(row['user']),
            'item': torch.tensor(row['item']),
            'rating': torch.tensor(row['stars'], dtype=torch.float),
            'timestamp': torch.tensor(row['timestamp'], dtype=torch.float)
        }
