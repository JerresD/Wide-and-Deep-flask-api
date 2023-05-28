import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load the data and perform preprocessing
db = pd.read_csv("Restaurants_Penang.csv")
db = db.dropna()
cuisine_types = db['cuisine_type'].str.get_dummies(sep=',')
db = pd.concat([db, cuisine_types], axis=1)
scaler = MinMaxScaler()
db[['price_level', 'rating', 'review_count']] = scaler.fit_transform(db[['price_level', 'rating', 'review_count']])


wide_features = list(cuisine_types.columns)
deep_features = ['price_level', 'rating', 'review_count']

@app.route("/")
def Home():
    return render_template("index.html")

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    data = request.json
    favorite_cuisine = data["Cuisine Type"]
    favorite_price_level = float(data["Price Level"])
    
    # favorite_cuisine = request.form.get("Cuisine Type")
    # favorite_price_level = float(request.form.get("Price Level"))

    user_price_level = (favorite_price_level - db['price_level'].min()) / (db['price_level'].max() - db['price_level'].min()) 
    filtered_restaurants = db[(db['price_level'] <= user_price_level) & (db[favorite_cuisine] == 1)]
 
    # Sort the filtered restaurants by predicted recommendation probability
    filtered_restaurants['Prediction'] = model.predict([filtered_restaurants[wide_features], filtered_restaurants[deep_features]])
    filtered_restaurants = filtered_restaurants.sort_values(by='Prediction', ascending=False)

    normalized_data = filtered_restaurants[['price_level', 'rating', 'review_count']]
    denormalized_data = scaler.inverse_transform(normalized_data)
    filtered_restaurants[['price_level', 'rating', 'review_count']] = denormalized_data

    # Get the top recommended restaurants
    top_recommendations = filtered_restaurants.head(5)

    # Prepare the response data
    response = {
        'top_recommendations': top_recommendations[['restaurant', 'cuisine_type', 'price_level', 'rating', 'review_count', 'Prediction']].to_dict(orient='records')
    }
    # response = {
    #         'cuisine_type': favorite_cuisine, 'price_level': favorite_price_level
    #     }
    return jsonify(response)
    #return(favorite_price_level)

if __name__ == '__main__':
    app.run()
