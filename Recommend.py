"""Content-based recommendation system optimized for web use.
Only top-N recommendations are saved per user for faster retrieval.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import joblib
import os
from tqdm import tqdm

# Đọc dữ liệu
books_df = pd.read_csv("books_data.csv")
ratings_df = pd.read_csv("Books_rating.csv")

# Làm sạch và tiền xử lý
books_df.fillna('', inplace=True)
books_df.columns = books_df.columns.str.strip()
ratings_df.columns = ratings_df.columns.str.strip()

# Chuyển hết về chữ thường để đồng nhất
books_df['Title'] = books_df['Title'].str.lower()
ratings_df['Title'] = ratings_df['Title'].str.lower()

# Tạo trường đặc trưng kết hợp
books_df['combined_features'] = (
    books_df['Title'] + ' ' +
    books_df['authors'].fillna('') + ' ' +
    books_df['description'].fillna('') + ' ' +
    books_df['categories'].fillna('')
)

# TF-IDF vectorizer
vectorizer_path = "vectorizer.pkl"
X_path = "tfidf_features.npz"

if os.path.exists(vectorizer_path) and os.path.exists(X_path):
    vectorizer = joblib.load(vectorizer_path)
    X = joblib.load(X_path)
else:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english',max_features=5000)
    X = vectorizer.fit_transform(books_df['combined_features'])
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(X, X_path)

# Ánh xạ title -> index để dùng sau
title_to_index = {title: i for i, title in enumerate(books_df['Title'])}
index_to_title = {i: title for i, title in enumerate(books_df['Title'])}  # Fixed here
 
# Gợi ý cho người dùng: lưu top-N recommendation
user_top_recommendations = {}

unique_users = ratings_df['User_id'].unique()[:1000]  # lấy 1000 người đầu tiên để giảm tải

for user_id in tqdm(unique_users):
    user_ratings = ratings_df[ratings_df['User_id'] == user_id]
    rated_titles = user_ratings['Title'].values

    # Chỉ giữ những sách có trong books_df
    book_indices = [title_to_index[title] for title in rated_titles if title in title_to_index]
    scores = user_ratings[user_ratings['Title'].isin(title_to_index.keys())]['review/score'].values

    if len(book_indices) < 1 or len(book_indices) != len(scores):
        continue

    model = Ridge(alpha=1.0)
    model.fit(X[book_indices], scores)
    predictions = model.predict(X)

    # Lưu top-N (ví dụ: top 10) recommendation cho user này
    top_indices = np.argsort(predictions)[-10:][::-1]  # top 10
    top_titles = [index_to_title[i] for i in top_indices if i in index_to_title]
    user_top_recommendations[user_id] = top_titles

# Lưu kết quả nhỏ gọn để dùng cho web
joblib.dump(user_top_recommendations, "user_top_recommendations.pkl", compress=3)
joblib.dump(title_to_index, "title_to_index.pkl")
joblib.dump(books_df, "books_df.pkl")

print("Training hoàn tất và đã lưu top-N recommendation cho từng user.")

from sklearn.metrics import mean_squared_error

# ===== Đánh giá mô hình =====

all_true = []
all_pred = []
precision_list = []

k = 5  # Precision@K

for user_id in tqdm(unique_users):
    user_ratings = ratings_df[ratings_df['User_id'] == user_id]
    rated_titles = user_ratings['Title'].values
    scores = user_ratings['review/score'].values

    # Chỉ giữ những sách có trong tập sách
    book_indices = [title_to_index[title] for title in rated_titles if title in title_to_index]
    filtered_scores = [score for title, score in zip(rated_titles, scores) if title in title_to_index]

    if len(book_indices) < 1 or len(book_indices) != len(filtered_scores):
        continue

    model = Ridge(alpha=100)
    model.fit(X[book_indices], filtered_scores)
    predictions = model.predict(X)

    # Lưu RMSE data
    pred_scores = model.predict(X[book_indices])
    all_true.extend(filtered_scores)
    all_pred.extend(pred_scores)

    # Precision@K
    predictions_scaled = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    top_k_indices = np.argsort(predictions_scaled)[-k:][::-1]
    top_k_titles = [index_to_title[i] for i in top_k_indices if i in index_to_title]

    # Xem người dùng đã đánh giá những sách nào trong top-K
    relevant_titles = set(user_ratings['Title'].values)
    hits = sum([1 for title in top_k_titles if title in relevant_titles])
    precision = hits / k
    precision_list.append(precision)

# Tính RMSE
rmse = np.sqrt(mean_squared_error(all_true, all_pred))
print(f"\n📊 RMSE của mô hình: {rmse:.4f}")

# Tính Precision@K trung bình
avg_precision_k = np.mean(precision_list)
print(f"🎯 Precision@{k} trung bình: {avg_precision_k:.4f}")
