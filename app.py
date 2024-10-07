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

class BrowsingHistory(db.Model):
    __tablename__ = 'BrowsingHistory'  # Specify the table name here
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    user = db.relationship('User', backref='browsing_history')


def truncate(text, length=20):
    return text if len(text) <= length else text[:length] + '...'

def log_browsing_history(user_id, product_id):
    # Create a new browsing history entry
    history_entry = BrowsingHistory(user_id=user_id, product_id=product_id)
    
    # Add to session and commit to save to the database
    db.session.add(history_entry)
    db.session.commit()

def get_browsing_history(user_id):
    return BrowsingHistory.query.filter_by(user_id=user_id).order_by(BrowsingHistory.timestamp.desc()).all()

def recommend_based_on_browsing(train_data, target_user_id, top_n=16):
    # Fetch the user's browsing history
    history = get_browsing_history(target_user_id)
    print("User's Browsing History:", history)  # Debugging line

    if not history:
        return pd.DataFrame()  # Optionally, provide some default recommendations

    # Extract the product IDs from the user's browsing history
    viewed_product_ids = [entry.product_id for entry in history]
    print("Viewed Product IDs:", viewed_product_ids)  # Debugging line
    viewed_products = train_data[train_data['ProductID'].isin(viewed_product_ids)]
    print("Viewed Products DataFrame:", viewed_products)  # Debugging line

    if not viewed_products.empty:
        last_viewed_product_name = viewed_products.iloc[-1]['Name']
        print("Last Viewed Product Name:", last_viewed_product_name)  # Debugging line
        content_based_rec = get_similar_product_details(train_data, last_viewed_product_name)

        return content_based_rec[['ProductID', 'Name', 'discounted_price', 'Rating', 'Description', 'ImageURL', 'Rating_Count']].sort_values(by='Rating', ascending=False).head(top_n)
    
    return pd.DataFrame()  # Return empty if no viewed products


def get_similar_product_details(train_data, item_name):   
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Name'])
    
    item_vector = tfidf_vectorizer.transform([item_name])
    cosine_sim = cosine_similarity(item_vector, tfidf_matrix)
    
    similar_items = list(enumerate(cosine_sim[0]))
    similar_items_sorted = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:17]
    similar_items_indexed = [i[0] for i in similar_items_sorted]
    
    # Include ProductID in the returned DataFrame
    similar_items_data = train_data.iloc[similar_items_indexed][['ProductID', 'Name', 'actual_price', 'Rating', 'Description', 'ImageURL', 'Rating_Count']]
    
    return similar_items_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=12):
    # Create the user-item matrix
    user_item_matrix = train_data.pivot_table(index='shorten_user_id', columns='ProductID', values='Rating', aggfunc='mean').fillna(0)

    # Calculate the user similarity matrix using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)

    # Find the index of the target user in the matrix
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    # Get the similarity scores for the target user
    user_similarities = user_similarity[target_user_index]

    # Sort the users by similarity in descending order (excluding the target user)
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    # Generate recommendations based on similar users
    recommended_items = set()  # Use a set to avoid duplicates

    for user_index in similar_users_indices:
        # Get items rated by the similar user but not by the target user
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)

        # Extract the item IDs of recommended items
        recommended_items.update(user_item_matrix.columns[not_rated_by_target_user])

        if len(recommended_items) >= top_n:
            break  # Stop when we have enough recommendations

    # Get the details of recommended items
    recommended_items_details = train_data[train_data['ProductID'].isin(recommended_items)][['ProductID', 'Name', 'discounted_price', 'Rating_Count', 'Description', 'ImageURL', 'Rating']]
    top_items = recommended_items_details.sort_values(by='Rating', ascending=False).head(top_n)

    return top_items

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

def hybrid_recommendations(train_data, target_user_id, item_name, top_n=16):
    # Get content-based recommendations
    content_based_rec = get_similar_product_details(train_data, item_name)

    # Get collaborative filtering recommendations
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data, target_user_id)

    # Add missing columns to collaborative_filtering_rec with default values
    if 'discounted_price' not in collaborative_filtering_rec.columns:
        collaborative_filtering_rec['discounted_price'] = 0  # or some default value

    # Ensure both recommendations are DataFrames with the same structure
    content_based_rec = content_based_rec[['ProductID', 'Name', 'discounted_price', 'Rating', 'Description', 'ImageURL', 'Rating_Count']]
    collaborative_filtering_rec = collaborative_filtering_rec[['Name', 'Rating_Count', 'Description', 'ImageURL', 'Rating', 'discounted_price']]

    # Merge and deduplicate the recommendations
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates(subset='Name')
    
    return hybrid_rec.head(top_n)



# Function to get similar products when clicking on a product (using ProductID)
def on_product_click(train_data, clicked_product_id):
    # Get the name of the clicked product using the ProductID
    clicked_product_name = train_data[train_data['ProductID'] == clicked_product_id]['Name'].values[0]
    
    # Call the function to get similar products based on the clicked product's name, excluding the clicked product
    return get_similar_product_details(train_data, clicked_product_name, exclude_id=clicked_product_id)

def get_random_user_id():
    random_row = train_data.sample(n=1)  # Get one random row
    return random_row['shorten_user_id'].values[0]

def get_product_price(product_id):
    # Retrieve the price based on product_id
    product = train_data[train_data['ProductID'] == product_id]
    if not product.empty:
        return product['discounted_price'].values[0]  # Adjust 'Price' to your actual column name for price
    return None

@app.route('/')
def index():
    # Create a list of random prices for each product
    price = [random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]) for _ in range(len(trending_products))]
    
    # Initialize browsing_recommendations as empty for guest users
    browsing_recommendations = pd.DataFrame()  # Assuming it's a DataFrame, adjust accordingly
    
    # Check if the user is logged in
    if 'username' in session:
        user_id = session.get('user_id')

        # Get recommendations for the logged-in user based on browsing history
        browsing_recommendations = recommend_based_on_browsing(train_data, user_id, top_n=12)

        # Get recommendations for the logged-in user
        recommended_items = collaborative_filtering_recommendations(train_data, user_id, top_n=12)

        return render_template('index.html', 
                               trending_products=trending_products[:12], 
                               truncate=truncate, 
                               prices=price, 
                               username=session['username'], 
                               user_id=user_id, 
                               recommended_items=recommended_items,
                               browsing_recommendations=browsing_recommendations)
    
    # If not logged in, pass empty browsing_recommendations
    return render_template('index.html', 
                           trending_products=trending_products[:12], 
                           truncate=truncate, 
                           prices=price)  # Ensure it's passed even for guests


# 'signin' Route
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username, password=password).first()
        
        if user:
            session['username'] = user.username
            session['user_id'] = user.id
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
        
        # Check if the user is logged in
        if 'username' in session:
            user_id = session.get('user_id')

            # Use hybrid recommendations for logged-in users
            hybrid_rec = hybrid_recommendations(train_data, user_id, prod, top_n=16)

            if hybrid_rec.empty:
                message = "No recommendations available for this product."
                return render_template('main.html', message=message, username=session['username'], user_id=user_id)
            else:
                return render_template('main.html', 
                                       content_based_rec=hybrid_rec, 
                                       truncate=truncate, username=session['username'], user_id=user_id,
                       prod=prod)
        else:
            # Use content-based recommendations for guest users
            content_based_rec = get_similar_product_details(train_data, prod)

            if content_based_rec.empty:
                message = "No recommendations available for this product."
                return render_template('main.html', message=message)
            else:
                return render_template('main.html', 
                                       content_based_rec=content_based_rec, 
                                       truncate=truncate,
                       prod=prod)


@app.route('/product/<product_id>')
def product_detail(product_id):
    # Retrieve the product details based on product_id
    product_details = train_data[train_data['ProductID'] == product_id]
    
    if product_details.empty:
        return "Product not found.", 404
    
    # Log the browsing history for the logged-in user
    if 'username' in session:
        user_id = session.get('user_id')
        username = session['username']  # Get the username from the session
        log_browsing_history(user_id, product_id)  # Log the viewing
    else:
        username = None  # Set username to None if not in session
    
    # Get the price
    discounted_price = get_product_price(product_id)
    
    # Call the function to get similar products
    similar_products = on_product_click(train_data, product_id)
    
    # Pass product details, similar products, and username to the template
    return render_template('product_detail.html', 
                           product=product_details.iloc[0], 
                           discounted_price=discounted_price,  
                           similar_products=similar_products,
                           username=username,  # Pass username to the template
                           truncate=truncate)


# Log out route
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the username from the session
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)