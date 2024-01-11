# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pickle
from flask import Flask, request, jsonify

# Load the data
data = pd.read_csv('Personalized_Locations.csv', encoding='latin-1')

# Define mapping functions for data preprocessing
# (Copy the functions you defined in your Colab notebook)

def map_district(value):
    district_mapping = {
        'Colombo': 1,
        'Gampaha': 2,
        'Mannar': 3,
        'Kilinochchi': 4,
        'Mulative': 5,
        'Jaffna': 6,
        'Vavuniya': 7,
        'Batticaloa': 8,
        'Trincomalee': 9,
        'Monaragala': 10,
        'Badulla': 11,
        'Hambanthota': 12,
        'Matara': 13,
        'Galle': 14,
        'Matale': 15,
        'Nuwara Eliya': 16,
        'Kandy': 17,
        'Polonnaruwa': 18,
        'Anuradhapura': 19,
        'Kegalle': 20,
        'Ratnapura': 21,
        'Puttalam': 22,
        'Kurunagala': 23,
        'Kalutara': 24,
        'Ampara': 25
    }
    return district_mapping.get(value, 0)

def map_budget(value):
    budget_mapping = {
        'High': 1,
        'Low': 2,
        'Free': 3,
        'Medium': 4
    }
    return budget_mapping.get(value, 0)

def map_type(value):
    type_mapping = {
        'Tourist Place': 1,
        'Religious Places': 2,
        'Historical landmark': 3
    }
    return type_mapping.get(value, 0)

def map_weather(value):
    weather_mapping = {
        'Mild': 1,
        'Dry': 2,
        'Tropical': 3,
        'Tropical monsoon Season': 4,
        'Tropical savanna Season': 5,
        'Tropical nature Season': 6,
        'Dry-monsoonal Season': 7,
        'Tropical rainforest Season': 8
    }
    return weather_mapping.get(value, 0)

def map_category(value):
    category_mapping = {
        'Temple': 1,
        'Garden': 2,
        'Fort': 3,
        'Forest': 4,
        'Wildlife Tourism Zones': 5,
        'Mountain areas': 6,
        'Beach': 7,
        'Water Fall': 8,
        'Lake': 9,
        'Rock': 10,
        'Museum': 11,
        'Park': 12,
        'Lighthouse': 13,
        'Outdoor': 14,
        'Island': 15,
        'Hindu Temple': 16,
    }
    return category_mapping.get(value, 0)


# Apply mapping functions to convert categorical columns to numeric
data['District'] = data['District'].apply(map_district)
data['Budget'] = data['Budget'].apply(map_budget)
data['Type'] = data['Type'].apply(map_type)
data['Weather Type '] = data['Weather Type '].apply(map_weather)
data['Category'] = data['Category'].apply(map_category)

# Encode the 'Name' column
name_encoder = LabelEncoder()
data['Name'] = name_encoder.fit_transform(data['Name'])

# Set the target variable 'y' to a numerical column in your dataset
y = data['Name']

# Split the data into features (X) and target (y)
X = data.drop('Name', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the trained Support Vector Regression (SVR) model from the pickle file
with open('SVR_Model.pkl', 'rb') as svr_file:
    svr = pickle.load(svr_file)

# Load the trained Random Forest Regressor model from the pickle file
with open('RandomForest_Model.pkl', 'rb') as rf_file:
    rf = pickle.load(rf_file)

# ... (Previous code for data preprocessing and model loading)

# Create a Flask application
app = Flask(__name__)

# Create a Flask route for receiving POST requests
@app.route('/predict/personalizedlocation', methods=['POST'])
def predict():
    try:
        data = request.json

        # Convert the received data to numeric values using the mapping functions
        data['District'] = map_district(data['District'])
        data['Budget'] = map_budget(data['Budget'])
        data['Type'] = map_type(data['Type'])
        data['Weather Type '] = map_weather(data['Weather Type '])
        data['Category'] = map_category(data['Category'])

        # Create a DataFrame from the received data
        sample_df = pd.DataFrame([data])

        # Ensure the column order matches X_train
        sample_df = sample_df[X_train.columns]

        # Scale the features of the sample data
        sample_scaled = scaler.transform(sample_df)

        # Make predictions using the trained models
        svr_prediction = svr.predict(sample_scaled)
        rf_prediction = rf.predict(sample_scaled)

        # Inverse transform the predictions to get the original 'Name' values
        svr_prediction = name_encoder.inverse_transform([int(svr_prediction)])
        rf_prediction = name_encoder.inverse_transform([int(rf_prediction)])

        response = {
            "SVR_Predicted_Name": svr_prediction[0],
            "Random_Forest_Predicted_Name": rf_prediction[0]
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5013)
