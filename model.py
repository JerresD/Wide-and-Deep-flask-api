import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

db = pd.read_csv("Restaurants_Penang.csv")
db = db.dropna()

# Apply the threshold to the rating and review_count columns to create the recommended column
rating_threshold = 4.0  
review_count_threshold = 1 
db["Recommended"] = ""
db['Recommended'] = (db['rating'] >= rating_threshold) & (db['review_count'] >= review_count_threshold)
db['Recommended'] = db['Recommended'].astype(int)  # convert boolean values to integers (0 or 1)

# One-hot encode the cuisine types
cuisine_types = db['cuisine_type'].str.get_dummies(sep=',')
db = pd.concat([db, cuisine_types], axis=1)

# normalize rating and review using MinMaxScaler
scaler = MinMaxScaler()
db[['price_level', 'rating', 'review_count']] = scaler.fit_transform(db[['price_level', 'rating', 'review_count']])

# Define the input features for the wide & deep part of the model
wide_features = list(cuisine_types.columns)
deep_features = ['price_level', 'rating', 'review_count']

# Split the data into training and testing sets
train_data, test_data = train_test_split(db, test_size=0.2)

# Define the wide and deep model
wide_input = tf.keras.layers.Input(shape=(len(wide_features),))
deep_input = tf.keras.layers.Input(shape=(len(deep_features),))
wide_layer = tf.keras.layers.Dense(1, activation='sigmoid')(wide_input)
deep_layer = tf.keras.layers.Dense(10, activation='relu')(deep_input)
deep_layer = tf.keras.layers.Dense(10, activation='relu')(deep_layer)
wide_deep_output = tf.keras.layers.concatenate([wide_layer, deep_layer])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(wide_deep_output)
model = tf.keras.models.Model(inputs=[wide_input, deep_input], outputs=output_layer)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([train_data[wide_features], train_data[deep_features]], train_data['Recommended'], epochs=10, batch_size=32, validation_split=0.2)


# Prompt the user for their favorite cuisine type and price level
favorite_cuisine = input("Enter your favorite cuisine type: ")
favorite_price_level = float(input("Enter your favorite price level (1-3): "))

# Normalize user input
user_price_level = (favorite_price_level - db['price_level'].min()) / (db['price_level'].max() - db['price_level'].min()) 

# Filter the restaurants based on user preferences
filtered_restaurants = db[(db['price_level'] <= favorite_price_level) & (db[favorite_cuisine] == 1)]

# Sort the filtered restaurants by predicted recommendation probability
filtered_restaurants['Prediction'] = model.predict([filtered_restaurants[wide_features], filtered_restaurants[deep_features]])
filtered_restaurants = filtered_restaurants.sort_values(by='Prediction', ascending=False)

# Display the top recommended restaurants
top_recommendations = filtered_restaurants.head(5)
print("Top", 5, "Recommended Restaurants:")
print(top_recommendations[['restaurant', 'cuisine_type', 'price_level', 'rating', 'review_count', 'Prediction']])

# Make pickle file of our model
#pickle.dump(model, open("model.pkl", "wb"))
#model.save('model.h5')
