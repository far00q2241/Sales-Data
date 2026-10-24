
import streamlit as st
import pandas as pd
import joblib
from datetime import date

# -----------------------------
# Load the trained model
# -----------------------------
model = joblib.load("sales_data_model.pkl")

# Get feature names stored in the trained XGBoost model
FEATURE_COLUMNS = model.get_booster().feature_names

if not FEATURE_COLUMNS:
    st.error("Model feature names are missing.")
    st.stop()

# -----------------------------
# App title
# -----------------------------
st.title("Sales Profit Prediction")
st.write("Enter order details to predict the expected profit.")

# -----------------------------
# User inputs
# -----------------------------
order_date = st.date_input(
    "Order Date",
    value=date.today()
)

quantity = st.number_input(
    "Quantity",
    min_value=1,
    value=1
)

discount = st.number_input(
    "Discount",
    min_value=0.0,
    value=0.0
)

discount_amount = st.number_input(
    "Discount Amount (INR)",
    min_value=0.0,
    value=0.0
)

shipping_days = st.number_input(
    "Shipping Days",
    min_value=0,
    value=3
)

satisfaction = st.selectbox(
    "Customer Satisfaction",
    [1, 2, 3, 4, 5],
    index=2
)

region = st.selectbox(
    "Region",
    ["East", "North", "South", "West"]
)

city = st.selectbox(
    "City",
    [
        "Bengaluru", "Bhopal", "Bhubaneswar", "Chandigarh",
        "Chennai", "Delhi", "Guwahati", "Hyderabad", "Indore",
        "Jaipur", "Kochi", "Kolkata", "Lucknow", "Mumbai",
        "Nagpur", "Patna", "Pune", "Raipur", "Surat"
    ]
)

segment = st.selectbox(
    "Segment",
    ["Corporate", "Small Business"]
)

channel = st.selectbox(
    "Channel",
    ["Online", "Retail Store"]
)

category = st.selectbox(
    "Category",
    ["Furniture", "Home & Kitchen", "Office Supplies", "Technology"]
)

subcategory = st.selectbox(
    "Subcategory",
    [
        "Appliances", "Binders", "Chairs", "Cleaning",
        "Cookware", "Decor", "Desks", "Haircare", "Laptops",
        "Monitors", "Makeup", "Paper", "Pens", "Personal Care",
        "Phones", "Printers", "Skincare", "Storage", "Tables"
    ]
)

returned = st.selectbox(
    "Returned",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Credit Card", "Debit Card", "Net Banking", "UPI"]
)

# -----------------------------
# Predict profit
# -----------------------------
if st.button("Predict Profit"):

    # Create input with the model's expected columns
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=FEATURE_COLUMNS
    )

    # Numerical features
    input_data["Quantity"] = quantity
    input_data["Discount"] = discount
    input_data["Discount_Amount_INR"] = discount_amount
    input_data["Shipping_Days"] = shipping_days
    input_data["Customer_Satisfaction"] = satisfaction

    # Date features
    input_data["Year"] = order_date.year
    input_data["Month"] = order_date.month
    input_data["Day"] = order_date.day

    # Categorical features
    categories = {
        "Region": region,
        "City": city,
        "Segment": segment,
        "Channel": channel,
        "Category": category,
        "Subcategory": subcategory,
        "Payment_Method": payment_method
    }

    for column, value in categories.items():
        feature = f"{column}_{value}"

        if feature in input_data.columns:
            input_data[feature] = 1

    # Returned feature
    if returned == "Yes" and "Returned_Yes" in input_data.columns:
        input_data["Returned_Yes"] = 1

    # Ensure the correct column order
    input_data = input_data[FEATURE_COLUMNS]

    # Make prediction
    try:
        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Profit: ₹{prediction:,.2f}")

    except Exception as error:
        st.error("Prediction failed. Please check the model features.")
        st.exception(error)
