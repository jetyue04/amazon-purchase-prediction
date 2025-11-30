import numpy as np
import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
val = pd.read_csv('data/val.csv')
items = pd.read_csv('data/items.csv')

# Item popularity based on training data
item_counts = train['parent_asin'].value_counts()#.to_dict()
popularity = np.log1p(item_counts) / np.log1p(item_counts).max()  # scaled 0-1

for k in [5, 10, 20, 50, 100, 500, 1000]:
    topK = popularity.head(k).index.tolist()
    hit = 0
    total = len(val)

    for _, row in val.iterrows():
        if row["parent_asin"] in topK:
            hit += 1

    hit_rate = hit / total
    print(f"Hit Rate @ {k}: {hit_rate}")

# Items purchased per user
user_to_items = train.groupby('user_id')['parent_asin'].apply(set).to_dict()

# Users who purchased each item
item_to_users = train.groupby('parent_asin')['user_id'].apply(set).to_dict()

def jaccard(item_a, item_b):
    users_a = item_to_users.get(item_a, set())
    users_b = item_to_users.get(item_b, set())
    if not users_a or not users_b:
        return 0.0
    return len(users_a & users_b) / len(users_a | users_b)

def smooth_jaccard(i, j, alpha=0.01, beta=0.01):
    users_i = item_to_users.get(i, set())
    users_j = item_to_users.get(j, set())
    if not users_i or not users_j:
        return 0.0
    return (len(users_i & users_j) + alpha) / (len(users_i | users_j) + beta)

# Score for user u and candidate item query_item
def score_user_item(u, query_item):
    if u not in user_to_items.keys():
        return 'LOL'
    items_u = user_to_items.get(u)
    sim_sum = sum(smooth_jaccard(query_item, item) for item in items_u)
    
    # normalize by number of items user bought to get mean similarity
    sim_mean = sim_sum / len(items_u) if items_u else 0.0
    
    # final score = popularity × similarity
    pop = popularity.get(query_item, 0)
    return pop * sim_mean

for user_item in val[['user_id', 'parent_asin']].itertuples(index=False):
    u, item = user_item
    sc = score_user_item(u, item)
    print(f"User: {u}, Item: {item}, Score: {sc}")