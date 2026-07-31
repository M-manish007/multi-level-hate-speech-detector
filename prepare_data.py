import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = os.path.join("data", "all_data.csv")
TRAIN_CLEAN_PATH = os.path.join("data", "train_clean.csv")
VAL_CLEAN_PATH = os.path.join("data", "val_clean.csv")

def preprocess():
    print("Loading and filtering dataset in memory-efficient chunks...")
    
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Could not find '{RAW_DATA_PATH}'. Make sure 'all_data.csv' is inside the 'data' folder!")

    # Load data in chunks of 50,000 rows to prevent RAM overload
    chunk_list = []
    target_samples = 30000
    collected = 0

    for chunk in pd.read_csv(RAW_DATA_PATH, usecols=['comment_text', 'split', 'toxicity'], chunksize=50000):
        # Filter for train split
        chunk_train = chunk[chunk['split'] == 'train'].copy()
        chunk_train = chunk_train.dropna(subset=['comment_text', 'toxicity'])
        
        if not chunk_train.empty:
            # Convert toxicity score to binary
            chunk_train['label'] = (chunk_train['toxicity'] >= 0.5).astype(int)
            chunk_list.append(chunk_train[['comment_text', 'label']])
            collected += len(chunk_train)
            
        # Stop loading more chunks once we have enough data
        if collected >= target_samples * 2:
            break

    # Combine collected chunks
    df_clean = pd.concat(chunk_list, ignore_index=True)
    
    # Sample exact target count (30,000)
    sample_df = df_clean.sample(n=min(target_samples, len(df_clean)), random_state=42)
    
    train, val = train_test_split(
        sample_df,
        test_size=0.15,
        random_state=42,
        stratify=sample_df['label']
    )
    
    train.to_csv(TRAIN_CLEAN_PATH, index=False)
    val.to_csv(VAL_CLEAN_PATH, index=False)
    
    print("Data preprocessing complete!")
    print(f"Saved Train: {len(train)} rows -> {TRAIN_CLEAN_PATH}")
    print(f"Saved Val:   {len(val)} rows -> {VAL_CLEAN_PATH}")

if __name__ == "__main__":
    preprocess()