from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sqlite3
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask import abort

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///book_recommendations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    book_title = db.Column(db.String(200), nullable=False)
    score = db.Column(db.Float, nullable=False)

# Load pre-trained model and data
try:
    books_df = joblib.load("books_df.pkl")
    title_to_index = joblib.load("title_to_index.pkl")
    user_recommendations = joblib.load("user_top_recommendations.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    X = joblib.load("tfidf_features.npz")
    
    # Ensure books_df has avg_rating column for fallback recommendations
    if 'avg_rating' not in books_df.columns:
        # Add a placeholder average rating if it doesn't exist
        books_df['avg_rating'] = 3.5
except Exception as e:
    print(f"Error loading model files: {e}")
    books_df = pd.DataFrame()
    title_to_index = {}
    user_recommendations = {}
    vectorizer = None
    X = None

# Initialize database
with app.app_context():
    db.create_all()
    
    # Initialize with some data if empty
    if User.query.count() == 0:
        admin = User(username="admin")
        admin.set_password("admin")
        db.session.add(admin)
        db.session.commit()

@app.route('/')
def index():
    return render_template('index.html', load_popular=True)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'message': 'Invalid username or password'})
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'})
        
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        return jsonify({'success': True})
    
    return render_template('register.html')

@app.route('/book/<title>')
def book_detail(title):
    # Normalize tiêu đề
    normalized_title = title.lower().strip()
    book_match = books_df[books_df['Title'] == normalized_title]
    
    if book_match.empty:
        return abort(404, description="Book not found.")
    
    book = book_match.iloc[0]
    return render_template('book_detail.html', book=book)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    
    # Simple search in books_df - increased from 10 to 30 results
    matches = books_df[books_df['Title'].str.contains(query.lower())]
    # Include previewLink in search results
    columns_to_include = ['Title', 'authors', 'description']
    if 'previewLink' in books_df.columns:
        columns_to_include.append('previewLink')
    if 'image' in books_df.columns:
        columns_to_include.append('image')
    
    results = matches[columns_to_include].head(30).to_dict('records')
    return jsonify(results)

@app.route('/rate', methods=['POST'])
def rate_book():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please log in first'})
    
    data = request.get_json()
    book_title = data.get('title')
    score = data.get('score')
    
    if not book_title or not score:
        return jsonify({'success': False, 'message': 'Missing data'})
    
    # Save rating to database
    rating = Rating(
        user_id=session['user_id'],
        book_title=book_title.lower(),
        score=float(score)
    )
    db.session.add(rating)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/recommendations', methods=['GET'])
def recommendations():
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Please log in first'})

        user_id = session['user_id']
        user_ratings = Rating.query.filter_by(user_id=user_id).all()

        if len(user_ratings) < 3:
            # Increased from 10 to 30 popular books
            popular_books = books_df.sort_values('avg_rating', ascending=False).head(31).iloc[1:]
            results = []
            for _, book in popular_books.iterrows():
                book_data = {
                    'title': book['Title'],
                    'authors': book.get('authors', 'Unknown'),
                    'description': str(book.get('description', 'No description available'))
                }
                # Add previewLink if available
                if 'previewLink' in book and pd.notna(book['previewLink']):
                    book_data['previewLink'] = book['previewLink']
                if 'image' in book and pd.notna(book['image']):
                    book_data['image'] = book['image']
                results.append(book_data)
            return jsonify({'success': True, 'recommendations': results, 'message': 'Based on popular books'})

        rated_titles = [rating.book_title for rating in user_ratings]
        scores = [rating.score for rating in user_ratings]
        book_indices = [title_to_index.get(title) for title in rated_titles]
        book_indices = [idx for idx in book_indices if idx is not None]

        if not book_indices:
            # Increased from 10 to 30 popular books
            popular_books = books_df.sort_values('avg_rating', ascending=False).head(10)
            results = []
            for _, book in popular_books.iterrows():
                book_data = {
                    'title': book['Title'],
                    'authors': book.get('authors', 'Unknown'),
                    'description': str(book.get('description', 'No description available'))
                }
                # Add previewLink if available
                if 'previewLink' in book and pd.notna(book['previewLink']):
                    book_data['previewLink'] = book['previewLink']
                if 'image' in book and pd.notna(book['image']):
                    book_data['image'] = book['image']
                results.append(book_data)
            return jsonify({'success': True, 'recommendations': results, 'message': 'Based on popular books'})

        filtered_scores = []
        for i, title in enumerate(rated_titles):
            if title in title_to_index:
                filtered_scores.append(scores[i])

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X[book_indices], filtered_scores)

        predictions = model.predict(X)
        rated_indices = set(book_indices)
        # Increased from 10 to 30 personalized recommendations
        top_indices = [i for i in np.argsort(predictions)[::-1] if i not in rated_indices][:10]

        recommendations = []
        for idx in top_indices:
            book = books_df.iloc[idx]
            book_data = {
                'title': book['Title'],
                'authors': book.get('authors', 'Unknown'),
                'description': str(book.get('description', 'No description available'))
            }
            # Add previewLink if available
            if 'previewLink' in book and pd.notna(book['previewLink']):
                book_data['previewLink'] = book['previewLink']
            if 'image' in book and pd.notna(book['image']):
                book_data['image'] = book['image']
            recommendations.append(book_data)

        return jsonify({'success': True, 'recommendations': recommendations})
    else:
        return render_template('recommendations.html')

@app.route('/popular-books')
def popular_books():
    try:
        # Lấy 21 cuốn sách phổ biến nhất, sau đó bỏ cuốn đầu tiên (từ 2 đến 21)
        popular_books = books_df.sort_values('avg_rating', ascending=False).head(21).iloc[1:]
        results = []
        
        for _, book in popular_books.iterrows():
            book_data = {
                'Title': book['Title'],
                'authors': book.get('authors', 'Unknown'),
                'description': str(book.get('description', 'No description available'))
            }
            # Add previewLink if available
            if 'previewLink' in book and pd.notna(book['previewLink']):
                book_data['previewLink'] = book['previewLink']
            if 'image' in book and pd.notna(book['image']):
                book_data['image'] = book['image']
            results.append(book_data)
            
        return jsonify(results)
        
    except Exception as e:
        print(f"Error getting popular books: {e}")
        return jsonify([])
if __name__ == '__main__':
    app.run(debug=True)