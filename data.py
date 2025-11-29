import json
import gzip
import pandas as pd
import numpy as np
import os

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

def prep_data(review_file, meta_file, save=True):
    # Load JSONL files
    reviews = to_df(review_file)
    items = to_df(meta_file)

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

if __name__ == "__main__":
    # Default file names
    review_file = "All_Beauty.jsonl.gz"
    meta_file = "meta_All_Beauty.jsonl.gz"
    
    print(f"Processing review file: {review_file}")
    print(f"Processing meta file: {meta_file}")
    
    train_df, val_df, test_df, items = prep_data(review_file, meta_file, save=True)
    
    print("\nData processing complete!")
    print(f"Output files saved to data/ directory:")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")
    print("  - items.csv")
