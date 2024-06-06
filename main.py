import pandas as pd
import streamlit as st
import datetime
import streamlit as st
import pickle
import sklearn

st.header('Cars24 Price Prediction App', divider='rainbow')

# df = pd.read_csv('cars24-car-price.csv')

# st.dataframe(df)

col1, col2, col3 = st.columns(3)

with col1:
    fuel_type = st.selectbox(
    "Select Fuel Type: ",
    ("Petrol", "Diesel", "CNG", "Electric", "LPG")
    )

with col2:
    transmission_type = st.selectbox(
    "Select Transmission Type: ",
    ("Manual", "Automatic")
    )

with col3:
    seats = st.selectbox(
    "Select no. of seats: ", [4, 5, 6, 7, 8]
    )

engine = st.slider("Set the Engine capacity", 600, 5000, step = 100)

encode_dict = {
    "fuel_type": {
        "Diesel" : 1,
        "Petrol" : 2,
        "CNG" : 3,
        "LPG" : 4,
        "Electric" : 5
    },

    "transmission_type": {
        "Manual" : 1,
        "Automactic" : 2
    }
}


def model_pred(fuel_encoded, transmission_encoded, seats, engine):
    with open("car_pred_model", "rb") as file:
        reg_model = pickle.load(file)
        input_features = [[2012.0, 1, 120000, fuel_encoded, transmission_encoded, 19.7, engine, 46.3, seats]]

        return reg_model.predict(input_features)


if st.button("Predict", type = "primary"):
    fuel_encoded = encode_dict['fuel_type'][fuel_type]
    transmission_encoded = encode_dict['transmission_type'][transmission_type]

    price = model_pred(fuel_encoded, transmission_encoded, seats, engine)
    st.text("Predicted Price is " + str(price))