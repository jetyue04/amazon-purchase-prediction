import json
import pandas as pd
import numpy as np

def to_df(file):
    rows = []
    with open(file, 'r') as fp:
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
        train_df.to_csv("data/train.csv", index=False)
        val_df.to_csv("data/val.csv", index=False)
        test_df.to_csv("data/test.csv", index=False)
        items.to_csv("data/items.csv", index=False)

    return train_df, val_df, test_df, items
