import json
import gzip
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def to_df(file):
    rows = []
    # Handle both .gz and regular files
    if file.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    with open_func(file, mode) as fp:
        for line in fp:
            rows.append(json.loads(line.strip()))
    df = pd.DataFrame(rows)
    return df

def prep_data(review_files, meta_files, save=True):
    # Load JSONL files
    review_dfs = [to_df(f) for f in review_files]
    reviews = pd.concat(review_dfs, ignore_index=True)

    # Load and concatenate all meta files
    item_dfs = [to_df(f) for f in meta_files]
    items = pd.concat(item_dfs, ignore_index=True)

    # Time-based split
    cutoff = reviews["timestamp"].quantile(0.80)
    train_df = reviews[reviews["timestamp"] < cutoff].copy()
    test_df  = reviews[reviews["timestamp"] >= cutoff].copy()

    val_cutoff = test_df["timestamp"].quantile(0.5)
    val_df  = test_df[test_df["timestamp"] < val_cutoff].copy()
    test_df = test_df[test_df["timestamp"] >= val_cutoff].copy()

    # Label positives
    train_df["label"] = 1
    val_df["label"]   = 1
    test_df["label"]  = 1

    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print(f"Items: {items.shape}")

    # Save to CSV
    if save:
        os.makedirs("data", exist_ok=True)
        train_df.to_csv("data/train.csv", index=False)
        val_df.to_csv("data/val.csv", index=False)
        test_df.to_csv("data/test.csv", index=False)
        items.to_csv("data/items.csv", index=False)

    return train_df, val_df, test_df, items

def create_evaluation_dataset(val_df, train_df, num_negatives=99, seed=42):
    np.random.seed(seed)
    
    # Get all items and user purchase history from training
    all_items = set(train_df['parent_asin'].unique())
    user_to_items = train_df.groupby('user_id')['parent_asin'].apply(set).to_dict()
    
    eval_rows = []
    
    print(f"\nGenerating evaluation dataset with {num_negatives} negatives per positive...")
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Creating eval dataset"):
        user_id = row['user_id']
        true_item = row['parent_asin']
        
        # Add positive example
        eval_rows.append({
            'user_id': user_id,
            'item_id': true_item,
            'label': 1
        })
        
        # Get items to exclude (user's training items)
        exclude_items = user_to_items.get(user_id, set())
        exclude_items.add(true_item)  # Also exclude the current validation item
        
        # Sample negatives
        available_items = list(all_items - exclude_items)
        if len(available_items) > 0:
            n_samples = min(num_negatives, len(available_items))
            negatives = np.random.choice(available_items, size=n_samples, replace=False)
            
            # Add negative examples
            for neg_item in negatives:
                eval_rows.append({
                    'user_id': user_id,
                    'item_id': neg_item,
                    'label': 0
                })
    
    eval_df = pd.DataFrame(eval_rows)
    print(f"Created evaluation dataset: {len(eval_df)} rows")
    print(f"  Positives: {eval_df['label'].sum()}")
    print(f"  Negatives: {(eval_df['label'] == 0).sum()}")
    
    return eval_df

if __name__ == "__main__":
    # Default file names
    review_file = ["All_Beauty.jsonl.gz"]
    meta_file = ["meta_All_Beauty.jsonl.gz"]
    
    print(f"Processing review file: {review_file}")
    print(f"Processing meta file: {meta_file}")
    
    # Prepare train/val/test splits
    train_df, val_df, test_df, items = prep_data(review_file, meta_file, save=True)
    
    # Create evaluation dataset with negatives for validation set
    eval_val_df = create_evaluation_dataset(val_df, train_df, num_negatives=99, seed=1)
    eval_val_df.to_csv("data/eval_val.csv", index=False)

    eval_test_df = create_evaluation_dataset(test_df, train_df, num_negatives=99, seed=2)
    eval_test_df.to_csv("data/eval_test.csv", index=False)
    
    print("\nData processing complete!")
    print(f"Output files saved to data/ directory:")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")
    print("  - items.csv")
    print("  - eval_val.csv")
    print("  - eval_test.csv")