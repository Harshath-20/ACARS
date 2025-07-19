# 🧠 ACARS – Advanced Context-Aware Recommender System

ACARS is a next-generation recommender system that combines:
- Transformer-based self-attention
- Contrastive self-supervised learning
- Multi-task learning (MTL)
- Reinforcement learning (optional Q-learning)

It is built using the **Yelp Open Dataset** for product and service recommendations.

---

## 📂 Project Structure

```
ACARS-Project/
├── acars/                   # Core model code (model, training, evaluation)
├── scripts/                 # Data conversion + negative sample generation
├── notebooks/               # Exploratory Data Analysis (EDA)
├── datasets/                # Yelp .json files
├── csv_outputs/             # Optional CSV files for inspection
├── requirements.txt         # Python dependencies
├── README.md                # You’re reading it!
```

---

## ⚙️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ACARS-Project.git
cd ACARS-Project
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

Place these files in the `datasets/` folder (from the Yelp Open Dataset(you can download from online file size is too big I can't upload it here):

- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_user.json`
- `yelp_academic_dataset_business.json`

(You can optionally include `checkin.json` and `tip.json` for future improvements.)

---

## 🚀 How to Run the Project

### ▶️ Train the ACARS Model:
```bash
python acars/train.py
```

This will:
- Train the model using Transformer + Contrastive + MTL
- Save weights to `acars_model.pth`

### 📊 Evaluate the Model:
```bash
python acars/evaluate.py
```

This prints:
- RMSE for rating prediction
- Precision@1 for CTR
- AUC for click prediction

---

## 🧪 Optional: Preprocess or Inspect Data

### Convert Yelp JSON to CSV:
```bash
python scripts/convert_yelp_to_csv.py
```

### Generate Negative Samples for Contrastive Learning:
```bash
python scripts/generate_negative_samples.py
```

---

## 📈 Explore Dataset in Notebook

Open the Jupyter notebook:
```bash
jupyter notebook notebooks/explore_yelp_data.ipynb
```

---

## 📌 Configuration

All paths, hyperparameters, and options are in:
```bash
acars/config.py
```

---

## 👨‍💻 Author

Built by [Harshath V, Pasumarti Vamsi Krishna]  
For research, academic evaluation, and production-ready recommendation systems.

---

## 📜 License

MIT License
