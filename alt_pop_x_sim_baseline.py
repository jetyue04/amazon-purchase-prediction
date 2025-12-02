import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# 1. LOAD DATA
# ============================================================================
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/val.csv')
items = pd.read_csv('data/items.csv')

print(f"Train: {len(train)} interactions")
print(f"Val: {len(val)} interactions")
print(f"Items: {len(items)} items")

# ============================================================================
# 2. BUILD MODEL COMPONENTS
# ============================================================================

# Compute item popularity from training data
item_counts = train['parent_asin'].value_counts()
popularity = np.log1p(item_counts) / np.log1p(item_counts).max()
all_items = train['parent_asin'].unique()

# Build user-item and item-user mappings
user_to_items = train.groupby('user_id')['parent_asin'].apply(set).to_dict()
item_to_users = train.groupby('parent_asin')['user_id'].apply(set).to_dict()

print(f"\nUnique users in train: {len(user_to_items)}")
print(f"Unique items in train: {len(all_items)}")

# ============================================================================
# 3. PREPROCESS VALIDATION DATA WITH NEGATIVES
# ============================================================================

def create_evaluation_dataset(val_df, all_items, user_to_items, num_negatives=99, seed=42):
    """
    Create evaluation dataset with negatives.
    
    Returns DataFrame with columns: user_id, item_id, label
    - label=1 for actual interactions
    - label=0 for negative samples
    """
    np.random.seed(seed)
    all_items_set = set(all_items)
    
    eval_rows = []
    
    print(f"\nGenerating evaluation dataset with {num_negatives} negatives per positive...")
    
    for idx, row in tqdm(val_df.iterrows()):
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
        available_items = list(all_items_set - exclude_items)
        negatives = np.random.choice(
            available_items,
            size=min(num_negatives, len(available_items)),
            replace=False
        )
        
        # Add negative examples
        for neg_item in negatives:
            eval_rows.append({
                'user_id': user_id,
                'item_id': neg_item,
                'label': 0
            })
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(val_df)} validation samples...")
    
    eval_df = pd.DataFrame(eval_rows)
    print(f"Created evaluation dataset: {len(eval_df)} rows")
    print(f"  Positives: {eval_df['label'].sum()}")
    print(f"  Negatives: {(eval_df['label'] == 0).sum()}")
    
    return eval_df

# Create evaluation dataset
eval_df = create_evaluation_dataset(val, all_items, user_to_items, num_negatives=99)

# ============================================================================
# 4. SIMILARITY FUNCTIONS
# ============================================================================

def smooth_jaccard(item_i, item_j, alpha=0.01, beta=0.01):
    """Compute smoothed Jaccard similarity between two items"""
    users_i = item_to_users.get(item_i, set())
    users_j = item_to_users.get(item_j, set())
    if not users_i or not users_j:
        return 0.0
    intersection = len(users_i & users_j)
    union = len(users_i | users_j)
    return (intersection + alpha) / (union + beta)

# ============================================================================
# 5. SCORING FUNCTION
# ============================================================================

def score_user_item(user_id, candidate_item):
    """Score a candidate item for a user"""
    if user_id not in user_to_items:
        return popularity.get(candidate_item, 0)
    
    user_items = user_to_items[user_id]
    similarities = [smooth_jaccard(candidate_item, item) for item in user_items]
    mean_similarity = np.mean(similarities) if similarities else 0.0
    pop_score = popularity.get(candidate_item, 0)
    
    return pop_score * mean_similarity

# ============================================================================
# 6. BATCH SCORING
# ============================================================================

def score_evaluation_dataset(eval_df):
    """Score all user-item pairs in the evaluation dataset"""
    print("\nScoring evaluation dataset...")
    scores = []
    
    for idx, row in eval_df.iterrows():
        score = score_user_item(row['user_id'], row['item_id'])
        scores.append(score)
        
        if (idx + 1) % 10000 == 0:
            print(f"  Scored {idx + 1}/{len(eval_df)} pairs...")
    
    eval_df['score'] = scores
    return eval_df

# Score the dataset
eval_df = score_evaluation_dataset(eval_df)

# ============================================================================
# 7. EVALUATION METRICS
# ============================================================================

def compute_metrics(eval_df, k_values=[5, 10, 20, 50, 100]):
    """
    Compute Hit Rate@K and MRR from scored evaluation dataset
    """
    results = {k: {'hits': 0} for k in k_values}
    reciprocal_ranks = []
    
    print("\nComputing metrics...")
    
    # Group by user_id (each group has 1 positive + N negatives)
    for user_id, group in eval_df.groupby('user_id'):
        # Sort by score descending
        ranked = group.sort_values('score', ascending=False)
        
        # Get ranked item list
        ranked_items = ranked['item_id'].values
        ranked_labels = ranked['label'].values
        
        # Find position of positive item (1-indexed)
        positive_positions = np.where(ranked_labels == 1)[0]
        
        if len(positive_positions) == 0:
            # No positive in this group (shouldn't happen)
            continue
        
        # Take first positive's rank
        true_rank = positive_positions[0] + 1
        
        # MRR
        reciprocal_ranks.append(1.0 / true_rank)
        
        # Hit Rate@K
        for k in k_values:
            if true_rank <= k:
                results[k]['hits'] += 1
    
    # Compute final metrics
    num_users = len(eval_df.groupby('user_id'))
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total users evaluated: {num_users}")
    print()
    
    for k in k_values:
        hit_rate = results[k]['hits'] / num_users
        print(f"Hit Rate@{k:3d}: {hit_rate:.4f}")
    
    mrr = np.mean(reciprocal_ranks)
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")
    print("="*60)
    
    return results, mrr

# ============================================================================
# 8. RUN EVALUATION
# ============================================================================

results, mrr = compute_metrics(eval_df, k_values=[5, 10, 20, 50, 100])