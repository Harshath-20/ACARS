import json
import pandas as pd
import os

def json_to_csv(input_path, output_path, max_lines=None):
    data = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    os.makedirs("csv_outputs", exist_ok=True)

    # Convert key Yelp files
    json_to_csv("datasets/yelp_academic_dataset_review.json", "csv_outputs/reviews.csv", max_lines=50000)
    json_to_csv("datasets/yelp_academic_dataset_user.json", "csv_outputs/users.csv", max_lines=50000)
    json_to_csv("datasets/yelp_academic_dataset_business.json", "csv_outputs/businesses.csv", max_lines=50000)
