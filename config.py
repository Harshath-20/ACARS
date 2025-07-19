# acars/config.py

# Dataset paths
REVIEW_PATH = "datasets/yelp_academic_dataset_review.json"
USER_PATH = "datasets/yelp_academic_dataset_user.json"
BUSINESS_PATH = "datasets/yelp_academic_dataset_business.json"

# Training parameters
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3

# Model parameters
EMBEDDING_DIM = 32
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4

# Contrastive Learning
TEMPERATURE = 0.5
CL_WEIGHT = 0.1

# Evaluation
MODEL_SAVE_PATH = "acars_model.pth"
