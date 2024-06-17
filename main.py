import pandas as pd
import streamlit as st
import datetime
import streamlit as st
import pickle
import sklearn

st.header('Cars24 Price Prediction App', divider='rainbow')

# df = pd.read_csv('cars24-car-price.csv')

# st.dataframe(df)

col1, col2 = st.columns(2)

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

col3, col4 = st.columns(2)

with col3:
    seats = st.selectbox(
    "Select no. of seats: ", [2, 3, 4, 5, 6, 7, 8, 9]
    )

with col4:
    seller_type = st.selectbox(
        "Select Seller Type", ("Individual", "Dealer")
    )

st.divider()

col5, col6 = st.columns(2)

with col5:
    year = st.slider(
        "Select the year of purchase", 1995, 2024, step = 1
    )

with col6:
    engine = st.slider(
        "Set the Engine capacity", 600, 5000, step = 100
    )

km_driven = st.slider("Select the km driven", 0, 200000, step = 500)


col7, col8 = st.columns(2)

with col7:
    mileage = st.slider(
        "Select the mileage", 0.0, 40.0, step = 0.5
    )

with col8:
    max_power = st.slider(
        "Select the max power", 50.0, 500.0, step = 5.0
    )


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
    },

    "seller_type": {
        "Individual" : 1,
        "Dealer" : 2
    }
}


def model_pred(year, seller_encoded,km_driven, fuel_encoded, transmission_encoded, mileage, engine, max_power, seats):
    with open("car_pred_model", "rb") as file:
        reg_model = pickle.load(file)
        input_features = [[year, seller_encoded, km_driven, fuel_encoded, transmission_encoded, mileage, engine, max_power, seats]]

        return reg_model.predict(input_features)

st.divider()

if st.button("Predict", type = "primary"):
    fuel_encoded = encode_dict['fuel_type'][fuel_type]
    transmission_encoded = encode_dict['transmission_type'][transmission_type]
    seller_encoded = encode_dict['seller_type'][seller_type]

    price = model_pred(year, seller_encoded,km_driven, fuel_encoded, transmission_encoded, mileage, engine, max_power, seats) * (10**5)
    price = round(price[0])
    st.success("Predicted Price is INR " + str(price))