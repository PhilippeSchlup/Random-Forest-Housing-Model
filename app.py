from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define data path and load the dataset
HOUSING_PATH = "./datasets/housing/"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

# Calculate derived attributes directly
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["population_per_household"] = housing["population"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]

# Get medians for the columns we removed
median_total_rooms = housing["total_rooms"].median()
median_total_bedrooms = housing["total_bedrooms"].median()
median_population = housing["population"].median()
median_households = housing["households"].median()

# Drop unnecessary columns for the pipeline fit
housing_prepared = housing.drop(columns=["median_house_value", "total_rooms", "total_bedrooms", "population", "households"])

# Define numeric and categorical attributes
housing_num = housing_prepared.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# Define pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# Fit the pipeline
full_pipeline.fit(housing_prepared)

# Load the model
housing_model = joblib.load("models/final_random_forest_housing_model.pkl")

# Create Flask app
app = Flask(__name__)
CORS(app)

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.get_json()

    # Prepare input data
    input_data = {
        "longitude": float(data['longitude']),
        "latitude": float(data['latitude']),
        "housing_median_age": int(data['housing_median_age']),
        "median_income": float(data['median_income']),
        "rooms_per_household": float(data['rooms_per_household']),
        "population_per_household": float(data['population_per_household']),
        "bedrooms_per_room": float(data['bedrooms']) / float(data["rooms_per_household"]),
        "ocean_proximity": data['feature_selection']
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Transform input data using the pipeline
    X_example_prepared = full_pipeline.transform(input_df)
    
    # Convert transformed array to DataFrame and add missing columns
    columns = list(num_attribs) + list(full_pipeline.named_transformers_["cat"].get_feature_names_out(["ocean_proximity"]))
    X_example_prepared_df = pd.DataFrame(X_example_prepared, columns=columns)

    # Add placeholder columns for the missing features
    X_example_prepared_df["total_rooms"] = median_total_rooms
    X_example_prepared_df["total_bedrooms"] = median_total_bedrooms
    X_example_prepared_df["population"] = median_population
    X_example_prepared_df["households"] = median_households

    # Ensure order of columns matches what the model expects
    X_example_prepared_df = X_example_prepared_df[sorted(X_example_prepared_df.columns)]

    # Convert to numpy array for prediction
    X_example_prepared = X_example_prepared_df.to_numpy()

    # Make prediction
    prediction = housing_model.predict(X_example_prepared)
    
    # Send response
    return jsonify({"predicted_median_house_value": prediction[0]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
