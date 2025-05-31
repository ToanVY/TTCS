import joblib
import pandas as pd

# Load dữ liệu đã lưu
user_top_recommendations = joblib.load("user_top_recommendations.pkl")
books_df = joblib.load("books_df.pkl")


def recommend_books_for_user(user_id, top_n=10):
    if user_id not in user_top_recommendations:
        print(f"Không tìm thấy dự đoán cho user_id: {user_id}")
        return pd.DataFrame()

    top_titles = user_top_recommendations[user_id][:top_n]

    # Lọc sách tương ứng từ books_df và sắp xếp theo thứ tự xuất hiện trong top_titles
    recommended_books = books_df[books_df['Title'].isin(top_titles)].copy()
    recommended_books['title_rank'] = recommended_books['Title'].apply(lambda t: top_titles.index(t))
    recommended_books.sort_values(by='title_rank', inplace=True)

    return recommended_books[['Title', 'authors', 'categories', 'description']]


# In ra danh sách user có sẵn để test
print("Một số user_id mẫu có thể dùng để test:")
for i, user_id in enumerate(user_top_recommendations.keys()):
    print(f"{i + 1}. {user_id}")
    if i == 9:  # In 10 user đầu tiên
        break

# Ví dụ: Gợi ý cho user đầu tiên
sample_user = list(user_top_recommendations.keys())[0]
print(f"\nĐề xuất sách cho user: {sample_user}\n")

recommendations = recommend_books_for_user(sample_user)
print(recommendations.to_string(index=False))
