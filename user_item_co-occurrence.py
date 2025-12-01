import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")
items = pd.read_csv("data/items.csv")

all_reviews = pd.concat([train, val, test], ignore_index=True)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}, Items: {len(items)}")
print(f"Total reviews: {len(all_reviews)}")

class CooccurrenceBaseline:
    def __init__(self):
        self.item_cooccur = defaultdict(Counter)
        self.user_items = defaultdict(set)
        self.item_category = {}
        self.item_brand = {}
        self.item_popularity = Counter()
        
        # Text similarity
        self.item_texts = {}
        self.text_vectorizer = None
        self.text_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        # Review embeddings
        self.item_reviews = defaultdict(list)
        self.review_vectorizer = None
        self.review_matrix = None
        self.item_to_review_idx = {}
        self.review_idx_to_item = {}
        
    def fit(self, train_df, items_df, all_reviews_df):
        for _, row in train_df[train_df['label'] == 1].iterrows():
            user = row['user_id']
            item = row['parent_asin']
            self.user_items[user].add(item)
            self.item_popularity[item] += 1
        
        for _, row in all_reviews_df.iterrows():
            if 'text' in row and pd.notna(row['text']):
                self.item_reviews[row['parent_asin']].append(str(row['text']))
        
        for user, items_set in self.user_items.items():
            items_list = list(items_set)
            for i in range(len(items_list)):
                for j in range(i+1, len(items_list)):
                    item_i, item_j = items_list[i], items_list[j]
                    self.item_cooccur[item_i][item_j] += 1
                    self.item_cooccur[item_j][item_i] += 1
        
        for _, row in items_df.iterrows():
            item_id = row['parent_asin']
            
            if 'main_category' in row and pd.notna(row['main_category']):
                self.item_category[item_id] = row['main_category']
            if 'store' in row and pd.notna(row['store']):
                self.item_brand[item_id] = row['store']
            
            text = ''
            if 'title' in row and pd.notna(row['title']):
                text += str(row['title']) + ' '
            if 'description' in row and pd.notna(row['description']):
                text += str(row['description'])
            
            if text.strip():
                self.item_texts[item_id] = text.strip()
        
        self._build_text_similarity()
        self._build_review_embeddings()
    
    # TF-IDF text embeddings
    def _build_text_similarity(self):
        if not self.item_texts:
            return
        
        items = list(self.item_texts.keys())
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.idx_to_item = {i: item for i, item in enumerate(items)}
        
        # TF-IDF
        texts = [self.item_texts[item] for item in items]
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.text_matrix = self.text_vectorizer.fit_transform(texts)
    
    def _build_review_embeddings(self):
        if not self.item_reviews:
            return
        
        items_with_reviews = [item for item, reviews in self.item_reviews.items() if reviews]
        self.item_to_review_idx = {item: i for i, item in enumerate(items_with_reviews)}
        self.review_idx_to_item = {i: item for i, item in enumerate(items_with_reviews)}
        
        review_docs = []
        for item in items_with_reviews:
            combined_reviews = ' '.join(self.item_reviews[item])
            review_docs.append(combined_reviews)
        
        # TF-IDF on reviews
        self.review_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            max_df=0.8,
            min_df=2
        )
        self.review_matrix = self.review_vectorizer.fit_transform(review_docs)
    
    # Review embedding similarity
    def _review_similarity(self, item1, item2):
        if self.review_matrix is None:
            return 0.0
        
        idx1 = self.item_to_review_idx.get(item1)
        idx2 = self.item_to_review_idx.get(item2)
        
        if idx1 is None or idx2 is None:
            return 0.0
        
        sim = cosine_similarity(
            self.review_matrix[idx1:idx1+1],
            self.review_matrix[idx2:idx2+1]
        )[0, 0]
        
        return max(0.0, sim)
    
    def _text_similarity(self, item1, item2):
        if self.text_matrix is None:
            return 0.0
        
        idx1 = self.item_to_idx.get(item1)
        idx2 = self.item_to_idx.get(item2)
        
        if idx1 is None or idx2 is None:
            return 0.0
        
        sim = cosine_similarity(
            self.text_matrix[idx1:idx1+1], 
            self.text_matrix[idx2:idx2+1]
        )[0, 0]
        
        return max(0.0, sim)
    
    # Predict purchase score for (user, item)
    def predict_score(self, user_id, item_id):
        user_history = self.user_items.get(user_id, set())
        
        # Use item popularity if no user history
        if not user_history:
            return self.item_popularity.get(item_id, 0) / max(self.item_popularity.values())
        
        # If user has already bought item
        if item_id in user_history:
            return 0.0
        
        score = 0.0
        
        for past_item in user_history:
            item_score = 0.0
            
            # Co-occurrence score (weight: 3.0)
            cooc_count = self.item_cooccur[past_item].get(item_id, 0)
            if cooc_count > 0:
                item_score += cooc_count * 3.0
            
            # Same category (weight: 1.0)
            if (self.item_category.get(past_item) == self.item_category.get(item_id) 
                and self.item_category.get(item_id) is not None):
                item_score += 1.0
            
            # Same brand/store (weight: 1.0)
            if (self.item_brand.get(past_item) == self.item_brand.get(item_id)
                and self.item_brand.get(item_id) is not None):
                item_score += 1.0
            
            # Text similarity (weight: 2.0)
            text_sim = self._text_similarity(past_item, item_id)
            item_score += text_sim * 2.0
            
            # Review embedding similarity (weight: 2.0)
            review_sim = self._review_similarity(past_item, item_id)
            item_score += review_sim * 2.0
            
            score += item_score
        
        score = score / len(user_history)
        return score
    
    # Predict scores for test set
    def predict(self, test_df):
        scores = []
        for idx, row in test_df.iterrows():
            if idx % 10000 == 0:
                print(f"  Processed {idx}/{len(test_df)}")
            score = self.predict_score(row['user_id'], row['parent_asin'])
            scores.append(score)
        return np.array(scores)
    
    # Evaluate ranking
    def evaluate_ranking(self, test_df, items_df, k_values=[5, 10, 20], neg_samples=99):
        all_items = set(items_df['parent_asin'].values)
        
        hit_rates = {k: [] for k in k_values}
        reciprocal_ranks = []
        
        for idx, pos_row in test_df.iterrows():
            user = pos_row['user_id']
            pos_item = pos_row['parent_asin']
            
            # User's purchase history
            user_history = self.user_items.get(user, set())
            
            # Sample negative items
            available_negs = list(all_items - user_history - {pos_item})
            if len(available_negs) < neg_samples:
                sampled_negs = available_negs
            else:
                sampled_negs = np.random.choice(available_negs, neg_samples, replace=False)
            
            pos_score = self.predict_score(user, pos_item)
            neg_scores = [self.predict_score(user, neg) for neg in sampled_negs]
            
            # Rank all items
            all_scores = [pos_score] + neg_scores
            all_items_list = [pos_item] + list(sampled_negs)
            ranked_indices = np.argsort(all_scores)[::-1]
            ranked_items = [all_items_list[i] for i in ranked_indices]
            pos_rank = ranked_items.index(pos_item) + 1
            
            # Hit Rate @ K
            for k in k_values:
                hit_rates[k].append(1.0 if pos_rank <= k else 0.0)
            
            reciprocal_ranks.append(1.0 / pos_rank)
        
        results = {}
        for k in k_values:
            results[f'hit_rate@{k}'] = np.mean(hit_rates[k])
        results['mrr'] = np.mean(reciprocal_ranks)
        
        return results

# Train model
model = CooccurrenceBaseline()
model.fit(train, items, all_reviews)

# Validation set
print("Validation Set:")
val_sample = val.sample(n=min(1000, len(val)), random_state=42)
val_metrics = model.evaluate_ranking(val_sample, items, k_values=[5, 10, 20], neg_samples=99)
for metric, value in val_metrics.items():
    print(f"{metric:15s}: {value:.4f}")

# Test set
print("Test Set:")
test_sample = test.sample(n=min(1000, len(test)), random_state=42)
test_metrics = model.evaluate_ranking(test_sample, items, k_values=[5, 10, 20], neg_samples=99)
for metric, value in test_metrics.items():
    print(f"{metric:15s}: {value:.4f}")