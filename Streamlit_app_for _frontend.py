import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st

# Load dataset
df = pd.read_csv('C:/Users/LENOVO/Downloads/cleaned_pakwheels.csv', index_col=0)
selected_features = ['model_year', 'mileage', 'engine_type', 'transmission', 'registered_in', 'assembly',
                     'engine_capacity', 'make', 'ev', 'has_ac', 'current_city', 'price']

# One-hot encode categorical features
df_selected = pd.get_dummies(df[selected_features], columns=['engine_type', 'transmission', 'registered_in', 'assembly', 'make', 'current_city'])

X = df_selected.drop('price', axis=1)
y = df_selected['price']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_models = 2

# Lists to store models, predictions, and metrics
models = []
predictions_list = []
mae_list = []
r2_list = []

# Bagging loop with different random seeds for each model
for i in range(num_models):
    # Reset the random seed for each iteration
    np.random.seed(i)

    # Create a new Random Forest Regressor inWstance with a unique seed
    model = RandomForestRegressor(n_estimators=100, random_state=i)  # You can adjust parameters as needed

    # Fit the model
    model.fit(X_train, y_train)

    # Store the model
    models.append(model)

# Streamlit app
def main():
    st.title("Car Price Prediction App")

    # User input form
    st.sidebar.header("User Input")

    model_year = st.sidebar.number_input("Model Year", min_value=1900, max_value=2023, value=2020)
    mileage = st.sidebar.number_input("Mileage (in km)")
    engine_type = st.sidebar.selectbox("Engine Type", df['engine_type'].unique())
    transmission = st.sidebar.selectbox("Transmission", df['transmission'].unique())
    registered_in = st.sidebar.selectbox("Registered In", df['registered_in'].unique())
    engine_capacity = st.sidebar.number_input("Engine Capacity", min_value=0.1, max_value=10.0, value=2.0)
    make = st.sidebar.selectbox("Make", df['make'].unique())
    ev = st.sidebar.selectbox("EV", df['ev'].unique())
    has_ac = st.sidebar.selectbox("Has AC", df['has_ac'].unique())
    current_city = st.sidebar.selectbox("Current City", df['current_city'].unique())

    # Create a dictionary from user input
    user_input = {
        'model_year': model_year,
        'mileage': mileage,
        'engine_type': engine_type,
        'transmission': transmission,
        'registered_in': registered_in,
        'assembly': 'Imported Cars',  # Assuming 'assembly' is always 'Imported Cars' for user input
        'engine_capacity': engine_capacity,
        'make': make,
        'ev': ev,
        'has_ac': has_ac,
        'current_city': current_city
    }

    # Make predictions with each model
    predictions_list = []
    for i, model in enumerate(models):
        # Create a dataframe from user input
        user_df = pd.DataFrame([user_input])

        # One-hot encode categorical features
        user_df = pd.get_dummies(user_df, columns=['engine_type', 'transmission', 'registered_in', 'assembly', 'make', 'current_city'])

        # Ensure that the user input dataframe has the same columns as the training data
        missing_cols = set(X.columns) - set(user_df.columns)
        for col in missing_cols:
            user_df[col] = 0

        # Reorder the columns to match the order of the training data
        user_df = user_df[X.columns]

        # Make prediction
        prediction = model.predict(user_df)
        predictions_list.append(prediction)

        st.subheader(f"Model {i + 1} Prediction")
        st.write(f"The predicted price is: PKR {prediction[0]:,.2f}")

    # Average predictions from all models
    ensemble_predictions = np.mean(predictions_list, axis=0)

    # Display ensemble prediction
    st.subheader("Ensemble Prediction")
    st.write(f'The ensemble predicted price is: PKR {ensemble_predictions[0]:,.2f}')

if __name__ == "__main__":
    main()


