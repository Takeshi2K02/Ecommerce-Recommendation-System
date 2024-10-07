from flask import Flask,request,render_template, redirect, url_for, flash
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

trending_products = pd.read_csv(".\\models\\popularProducts.csv")
train_data = pd.read_csv(".\\models\\train_data.csv")

app.secret_key = "ecomrecommendation"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecommerce_recommendation"
app.config['SQLALCHEMY_TRACK_MODIFICATION'] = False
db = SQLAlchemy(app)

# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def get_similar_product_details(train_data, item_name):   
    
    # Calculate cosine similarity based on the 'tags' column (product features)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Name'])
    
    item_vector = tfidf_vectorizer.transform([item_name])

    cosine_sim = cosine_similarity(item_vector,tfidf_matrix)
    
    # Get a list of similar items (including the target item)
    similar_items = list(enumerate(cosine_sim[0]))
    
    # Sort the items by similarity score in descending order and exclude the first item (itself)
    similar_items_sorted = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:17]

    # Get the indices of the similar items
    similar_items_indexed = [i[0] for i in similar_items_sorted]
    
    # Extract the details of the similar products
    similar_items_data = train_data.iloc[similar_items_indexed][['Name','Rating','Description','ImageURL','Rating_Count']]
    

    return similar_items_data


# List of predefined image URLs
# random_image_urls = [
#     "static/img_1.png",
#     "static/img_2.png",
#     "static/img_3.png",
#     "static/img_4.png",
#     "static/img_5.png",
# ]

#routes
@app.route("/")
def index():
    # Create a list of random image URLs for each product
    price = [random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]) for _ in range(len(trending_products))]

    return render_template('index.html', trending_products=trending_products.head(12), truncate=truncate, prices=price)

#routes
@app.route("/main")
def main():
    # Ensure content_based_rec is passed, even if it's empty
    content_based_rec = pd.DataFrame()  # Empty DataFrame for default
    return render_template('main.html', content_based_rec=content_based_rec)

#routes
@app.route("/index")
def indexredirectto():
    return render_template('index.html')

# @app.route("/signup", methods=['POST','GET'])
# def signup():
#     if request.method=='POST':
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        price = [random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]) for _ in range(len(trending_products))]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate, random_price=random.choice(price),
                               signup_message='User signed up successfully!')
    
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        print(request.form)  # Debugging line to see the submitted form data
        username = request.form.get('username')  # Using get to avoid KeyError
        password = request.form.get('password')
        
        if username is None or password is None:
            flash('Username or password not provided.', 'danger')
            return redirect(url_for('signin'))

        # Check if the username and password are correct
        user = Signin.query.filter_by(username=username, password=password).first()
        
        if user:
            flash('Sign in successful!', 'success')
            return redirect(url_for('welcome'))  # Redirect to a welcome page
        else:
            flash('Invalid username or password. Please try again.', 'danger')
            return redirect(url_for('signin'))  # Redirect back to sign-in page
    
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

if __name__=='__main__':
    app.run(debug=True)