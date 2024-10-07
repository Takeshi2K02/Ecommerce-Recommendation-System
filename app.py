from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random

app = Flask(__name__)

trending_products = pd.read_csv(".\\models\\popularProducts.csv")
train_data = pd.read_csv(".\\models\\train_data.csv")

app.secret_key = 'ecomrecommendation'

# Config for MySQL (make sure to replace with your MySQL credentials)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/ecommerce_recommendation'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define the User model
class User(db.Model):
    id = db.Column(db.String(100), primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

def truncate(text, length=20):
    return text if len(text) <= length else text[:length] + '...'

def get_similar_product_details(train_data, item_name):   
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Name'])
    
    item_vector = tfidf_vectorizer.transform([item_name])
    cosine_sim = cosine_similarity(item_vector, tfidf_matrix)
    
    similar_items = list(enumerate(cosine_sim[0]))
    similar_items_sorted = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:17]
    similar_items_indexed = [i[0] for i in similar_items_sorted]
    
    # Include ProductID in the returned DataFrame
    similar_items_data = train_data.iloc[similar_items_indexed][['ProductID', 'Name', 'discounted_price', 'Rating', 'Description', 'ImageURL', 'Rating_Count']]
    
    return similar_items_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to get similar product details based on a given item_name
def get_similar_product_details(train_data, item_name, exclude_id=None):   
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Name'])
    
    # Vectorize the provided item name
    item_vector = tfidf_vectorizer.transform([item_name])
    
    # Calculate cosine similarity between the input product and all products
    cosine_sim = cosine_similarity(item_vector, tfidf_matrix)
    
    # Enumerate and sort similar items by cosine similarity
    similar_items = list(enumerate(cosine_sim[0]))
    similar_items_sorted = sorted(similar_items, key=lambda x: x[1], reverse=True)
    
    # Optionally exclude the current product itself (by ProductID)
    if exclude_id is not None:
        similar_items_sorted = [item for item in similar_items_sorted if train_data.iloc[item[0]]['ProductID'] != exclude_id]
    
    # Select the top 16 similar items (excluding the first, which is the clicked item itself)
    similar_items_sorted = similar_items_sorted[:16]
    similar_items_indexed = [i[0] for i in similar_items_sorted]
    
    # Retrieve product details for the top similar items, including ProductID
    similar_items_data = train_data.iloc[similar_items_indexed][['ProductID', 'Name', 'discounted_price', 'Rating', 'Description', 'ImageURL', 'Rating_Count']]
    
    return similar_items_data

# Function to get similar products when clicking on a product (using ProductID)
def on_product_click(train_data, clicked_product_id):
    # Get the name of the clicked product using the ProductID
    clicked_product_name = train_data[train_data['ProductID'] == clicked_product_id]['Name'].values[0]
    
    # Call the function to get similar products based on the clicked product's name, excluding the clicked product
    return get_similar_product_details(train_data, clicked_product_name, exclude_id=clicked_product_id)

def get_random_user_id():
    random_row = train_data.sample(n=1)  # Get one random row
    return random_row['shorten_user_id'].values[0] 

@app.route('/')
def index():
    # Create a list of random prices for each product
    price = [random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]) for _ in range(len(trending_products))]
    
    # Check if the user is logged in
    if 'username' in session:
        return render_template('index.html', 
                               trending_products=trending_products[:12], 
                               truncate=truncate, 
                               prices=price, 
                               username=session['username'])
    
    # If not logged in, render the homepage without user-specific data
    return render_template('index.html', 
                           trending_products=trending_products[:12], 
                           truncate=truncate, 
                           prices=price)


# 'signin' Route
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username, password=password).first()
        
        if user:
            session['username'] = user.username
            return redirect(url_for('index'))
        else:
            return 'Invalid credentials'
    return render_template('signin.html')

# 'signup' Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if username or email already exists
        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()

        if existing_user:
            return 'Username already exists'
        elif existing_email:
            return 'Email already exists'
        
        # Initialize a variable to hold the random user ID
        random_user_id = None
        
        while True:
            # Get a random shorten_user_id to use as the user ID
            random_user_id = get_random_user_id()  # Fetch random shorten_user_id
            
            # Check if this ID already exists in the user table
            existing_user_id = User.query.filter_by(id=random_user_id).first()
            
            if not existing_user_id:
                break  # Exit loop if the ID is unique

        # Create new user with unique random_user_id
        new_user = User(id=random_user_id, username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('signin'))
    
    return render_template('signup.html')

#routes
@app.route("/main")
def main():
    # Ensure content_based_rec is passed, even if it's empty
    content_based_rec = pd.DataFrame()  # Empty DataFrame for default
    return render_template('main.html', content_based_rec=content_based_rec)

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        content_based_rec = get_similar_product_details(train_data, prod)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(trending_products['ImageURL']) for _ in range(len(content_based_rec))]

            print(content_based_rec)
            print(random_product_image_urls)

            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price))
        
# @app.route('/product/<product_id>')
# def product_detail(product_id):
#     # Logic to retrieve product details based on product_id
#     # You might simulate it or get it from train_data
#     product_details = train_data[train_data['ProductID'] == product_id]
    
#     if product_details.empty:
#         return "Product not found.", 404

#     return render_template('product_detail.html', product=product_details.iloc[0])

@app.route('/product/<product_id>')
def product_detail(product_id):
    # Retrieve the product details based on product_id
    product_details = train_data[train_data['ProductID'] == product_id]
    
    if product_details.empty:
        return "Product not found.", 404
    
    # Call the function to get similar products
    similar_products = on_product_click(train_data, product_id)
    
    # Pass product details and similar products to the template
    return render_template('product_detail.html', 
                           product=product_details.iloc[0], 
                           similar_products=similar_products)


# Log out route
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the username from the session
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)