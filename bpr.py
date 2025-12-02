import numpy as np
import pandas as pd
from collections import defaultdict
import random
import os
import pickle
from tqdm import tqdm

# ============================================================================
# 1. LOAD DATA
# ============================================================================
train_full = pd.read_csv('data/train.csv')
val_full = pd.read_csv('data/val.csv')
test = pd.read_csv('data/test.csv')
items = pd.read_csv('data/items.csv')

# Use full datasets
train = train_full  # Use full training set
val = val_full  # Use full validation set

print(f"Train: {len(train)} interactions")
print(f"Val: {len(val)} interactions")
print(f"Items: {len(items)} items")

# ============================================================================
# 2. HYPERPARAMETERS
# ============================================================================
embedding_dim = 50
learning_rate = 0.01
regularization = 0.01
num_epochs = 5
num_negatives = 5

# ============================================================================
# 3. BUILD USER-ITEM MAPPINGS
# ============================================================================
unique_users = train['user_id'].unique()
unique_items = train['parent_asin'].unique()

user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
idx_to_user = {idx: user for user, idx in user_to_idx.items()}
idx_to_item = {idx: item for item, idx in item_to_idx.items()}

num_users = len(user_to_idx)
num_items = len(item_to_idx)

print(f"\nUnique users: {num_users}")
print(f"Unique items: {num_items}")

# Build positive interactions (user-item pairs from training data)
positive_interactions = defaultdict(set)
for _, row in train.iterrows():
    u = row['user_id']
    i = row['parent_asin']
    if u in user_to_idx and i in item_to_idx:
        positive_interactions[user_to_idx[u]].add(item_to_idx[i])

# Convert to list for easier sampling
positive_pairs = []
for u_idx, items_set in positive_interactions.items():
    for i_idx in items_set:
        positive_pairs.append((u_idx, i_idx))

print(f"Positive interactions: {len(positive_pairs)}")

# ============================================================================
# 4. LOAD OR TRAIN BPR MODEL
# ============================================================================
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
user_emb_file = os.path.join(model_dir, "bpr_user_embeddings.npy")
item_emb_file = os.path.join(model_dir, "bpr_item_embeddings.npy")
mappings_file = os.path.join(model_dir, "bpr_mappings.pkl")

# Check if model exists and load it
if os.path.exists(user_emb_file) and os.path.exists(item_emb_file) and os.path.exists(mappings_file):
    print("\nLoading saved BPR model...")
    user_embeddings = np.load(user_emb_file)
    item_embeddings = np.load(item_emb_file)
    with open(mappings_file, 'rb') as f:
        loaded_mappings = pickle.load(f)
        user_to_idx = loaded_mappings['user_to_idx']
        item_to_idx = loaded_mappings['item_to_idx']
        idx_to_user = loaded_mappings['idx_to_user']
        idx_to_item = loaded_mappings['idx_to_item']
    num_users = user_embeddings.shape[0]
    num_items = item_embeddings.shape[0]
    print(f"Loaded model: {num_users} users, {num_items} items")
    print("Model loaded successfully! Skipping training.\n")
    skip_training = True
else:
    # Initialize embeddings
    np.random.seed(42)
    user_embeddings = np.random.normal(0, 0.1, (num_users, embedding_dim))
    item_embeddings = np.random.normal(0, 0.1, (num_items, embedding_dim))
    skip_training = False

# BPR Training
if not skip_training:
    print("\nTraining BPR model...")
    total_updates = len(positive_pairs) * num_negatives
    print(f"Total training updates per epoch: {total_updates:,}")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        random.shuffle(positive_pairs)
        
        for pair_idx, (u_idx, i_idx) in enumerate(tqdm(positive_pairs, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Sample negative items
            for _ in range(num_negatives):
                # Sample a negative item that user hasn't interacted with
                j_idx = random.randint(0, num_items - 1)
                while j_idx in positive_interactions[u_idx]:
                    j_idx = random.randint(0, num_items - 1)
                
                # Compute scores
                x_ui = np.dot(user_embeddings[u_idx], item_embeddings[i_idx])
                x_uj = np.dot(user_embeddings[u_idx], item_embeddings[j_idx])
                x_uij = x_ui - x_uj
                
                # Sigmoid of negative difference (for gradient)
                sigmoid_term = 1.0 / (1.0 + np.exp(x_uij))
                
                # Compute gradients
                grad_u = sigmoid_term * (item_embeddings[i_idx] - item_embeddings[j_idx]) - regularization * user_embeddings[u_idx]
                grad_i = sigmoid_term * user_embeddings[u_idx] - regularization * item_embeddings[i_idx]
                grad_j = -sigmoid_term * user_embeddings[u_idx] - regularization * item_embeddings[j_idx]
                
                # Update embeddings
                user_embeddings[u_idx] += learning_rate * grad_u
                item_embeddings[i_idx] += learning_rate * grad_i
                item_embeddings[j_idx] += learning_rate * grad_j
                
                # Compute loss (for monitoring)
                loss = -np.log(1.0 / (1.0 + np.exp(-x_uij)))
                total_loss += loss
        
        avg_loss = total_loss / (len(positive_pairs) * num_negatives)
        print(f"  Epoch {epoch + 1}/{num_epochs} complete - Average Loss: {avg_loss:.4f}")

    print("Training complete!")
    
    # Save the trained model
    print("\nSaving model...")
    np.save(user_emb_file, user_embeddings)
    np.save(item_emb_file, item_embeddings)
    with open(mappings_file, 'wb') as f:
        pickle.dump({
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'idx_to_user': idx_to_user,
            'idx_to_item': idx_to_item
        }, f)
    print(f"Model saved to {model_dir}/")
    print("  - bpr_user_embeddings.npy")
    print("  - bpr_item_embeddings.npy")
    print("  - bpr_mappings.pkl\n")
else:
    print("Using loaded model for evaluation.\n")

# ============================================================================
# 5. LOAD EVALUATION DATASET
# ============================================================================

# Load pre-computed evaluation dataset from data.py
print("\nLoading evaluation dataset from data/eval_val.csv...")
eval_df = pd.read_csv('data/eval_val.csv')
print(f"Loaded evaluation dataset: {len(eval_df)} rows")
print(f"  Positives: {eval_df['label'].sum()}")
print(f"  Negatives: {(eval_df['label'] == 0).sum()}")

# ============================================================================
# 6. SCORING FUNCTION
# ============================================================================

def score_user_item(user_id, candidate_item):
    """Score a candidate item for a user using BPR embeddings"""
    # Handle cold-start users/items
    if user_id not in user_to_idx:
        return 0.0
    if candidate_item not in item_to_idx:
        return 0.0
    
    u_idx = user_to_idx[user_id]
    i_idx = item_to_idx[candidate_item]
    
    # BPR score: dot product of user and item embeddings
    score = np.dot(user_embeddings[u_idx], item_embeddings[i_idx])
    return score

# ============================================================================
# 7. BATCH SCORING
# ============================================================================

def score_evaluation_dataset(eval_df):
    """Score all user-item pairs in the evaluation dataset - vectorized"""
    print("\nScoring evaluation dataset...")
    
    # Vectorized scoring - much faster!
    # Get indices for all user-item pairs at once
    user_indices = eval_df['user_id'].map(user_to_idx).fillna(-1).astype(int)
    item_indices = eval_df['item_id'].map(item_to_idx).fillna(-1).astype(int)
    
    # Compute scores vectorized
    scores = np.zeros(len(eval_df))
    valid_mask = (user_indices >= 0) & (item_indices >= 0)
    valid_user_idx = user_indices[valid_mask].values
    valid_item_idx = item_indices[valid_mask].values
    
    # Vectorized dot product for valid pairs
    if len(valid_user_idx) > 0:
        valid_scores = np.sum(user_embeddings[valid_user_idx] * item_embeddings[valid_item_idx], axis=1)
        scores[valid_mask] = valid_scores
    
    eval_df['score'] = scores
    return eval_df

# Score the dataset
eval_df = score_evaluation_dataset(eval_df)

# Add tiny random tiebreaker to break score ties (important for cold-start users)
np.random.seed(42)
eval_df['score_tiebreaker'] = np.random.rand(len(eval_df)) * 1e-10
eval_df['score_final'] = eval_df['score'] + eval_df['score_tiebreaker']

# ============================================================================
# 8. EVALUATION METRICS
# ============================================================================

def compute_metrics(eval_df, k_values=[5, 10, 20, 50, 100, 200]):
    """
    Compute Hit Rate@K and MRR from scored evaluation dataset
    """
    results = {k: {'hits': 0} for k in k_values}
    reciprocal_ranks = []
    
    # Track users in training vs cold-start
    users_in_training = set(user_to_idx.keys())
    users_in_training_count = 0
    cold_start_count = 0
    hits_in_training = {k: 0 for k in k_values}
    hits_cold_start = {k: 0 for k in k_values}
    
    print("\nComputing metrics...")
    
    # Group by user_id (each group has 1 positive + N negatives)
    for user_id, group in tqdm(eval_df.groupby('user_id'), desc="Computing metrics"):
        # Check if user is in training
        is_in_training = user_id in users_in_training
        if is_in_training:
            users_in_training_count += 1
        else:
            cold_start_count += 1
        
        # Sort by score descending (use score_final to break ties randomly)
        ranked = group.sort_values('score_final', ascending=False)
        
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
                if is_in_training:
                    hits_in_training[k] += 1
                else:
                    hits_cold_start[k] += 1
    
    # Compute final metrics
    num_users = len(eval_df.groupby('user_id'))
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total users evaluated: {num_users}")
    print(f"  Users in training: {users_in_training_count} ({users_in_training_count/num_users*100:.2f}%)")
    print(f"  Cold-start users: {cold_start_count} ({cold_start_count/num_users*100:.2f}%)")
    print()
    
    print("Overall Metrics:")
    for k in k_values:
        hit_rate = results[k]['hits'] / num_users
        print(f"Hit Rate@{k:3d}: {hit_rate:.4f}")
    
    mrr = np.mean(reciprocal_ranks)
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")
    
    # Breakdown by user type
    if users_in_training_count > 0:
        print(f"\nMetrics for users IN training ({users_in_training_count} users):")
        for k in k_values:
            hr = hits_in_training[k] / users_in_training_count
            print(f"  Hit Rate@{k:3d}: {hr:.4f}")
    
    if cold_start_count > 0:
        print(f"\nMetrics for COLD-START users ({cold_start_count} users):")
        for k in k_values:
            hr = hits_cold_start[k] / cold_start_count
            print(f"  Hit Rate@{k:3d}: {hr:.4f}")
    
    print("="*60)
    
    return results, mrr

# ============================================================================
# 9. RUN EVALUATION
# ============================================================================

results, mrr = compute_metrics(eval_df, k_values=[5, 10, 20, 50, 100, 200])

# Save results
output_file = os.path.join(model_dir, "bpr_evaluation_results.csv")
eval_df.to_csv(output_file, index=False)
print(f"\nEvaluation results saved to: {output_file}")
