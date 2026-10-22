
import streamlit as st
import pandas as pd
import joblib

from datetime import date

# Load trained model
model = joblib.load("sales_data_model.pkl")

# Model feature columns in training order
FEATURE_COLUMNS = [
    'Quantity', 'Discount', 'Discount_Amount_INR',
    'Shipping_Days', 'Customer_Satisfaction',
    'Year', 'Month', 'Day',
    'Region_East', 'Region_North', 'Region_South', 'Region_West',
    'City_Bengaluru', 'City_Bhopal', 'City_Bhubaneswar',
    'City_Chandigarh', 'City_Chennai', 'City_Delhi',
    'City_Guwahati', 'City_Hyderabad', 'City_Indore',
    'City_Jaipur', 'City_Kochi', 'City_Kolkata',
    'City_Lucknow', 'City_Mumbai', 'City_Nagpur',
    'City_Patna', 'City_Pune', 'City_Raipur', 'City_Surat',
    'Segment_Corporate', 'Segment_Small Business',
    'Channel_Online', 'Channel_Retail Store',
    'Category_Furniture', 'Category_Home & Kitchen',
    'Category_Office Supplies', 'Category_Technology',
    'Subcategory_Appliances', 'Subcategory_Binders',
    'Subcategory_Chairs', 'Subcategory_Cleaning',
    'Subcategory_Cookware', 'Subcategory_Decor',
    'Subcategory_Desks', 'Subcategory_Haircare',
    'Subcategory_Laptops', 'Subcategory_Monitors',
    'Subcategory_Makeup', 'Subcategory_Paper',
    'Subcategory_Pens', 'Subcategory_Personal Care',
    'Subcategory_Phones', 'Subcategory_Printers',
    'Subcategory_Skincare', 'Subcategory_Storage',
    'Subcategory_Tables', 'Returned_Yes',
    'Payment_Method_Credit Card', 'Payment_Method_Debit Card',
    'Payment_Method_Net Banking', 'Payment_Method_UPI'
]

# App title
st.title("Sales Profit Prediction")
st.write("Enter order details to predict the expected profit.")

# User inputs
order_date = st.date_input("Order Date", value=date.today())

quantity = st.number_input("Quantity", min_value=1, value=1)
discount = st.number_input("Discount", min_value=0.0, value=0.0)
discount_amount = st.number_input(
    "Discount Amount (INR)", min_value=0.0, value=0.0
)
shipping_days = st.number_input(
    "Shipping Days", min_value=0, value=3
)
satisfaction = st.slider(
    "Customer Satisfaction", 1, 5, 3
)

region = st.selectbox(
    "Region", ["East", "North", "South", "West"]
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
    "Segment", ["Corporate", "Small Business"]
)

channel = st.selectbox(
    "Channel", ["Online", "Retail Store"]
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

returned = st.selectbox("Returned", ["No", "Yes"])

payment_method = st.selectbox(
    "Payment Method",
    ["Credit Card", "Debit Card", "Net Banking", "UPI"]
)

# Predict profit
if st.button("Predict Profit"):

    # Create a row with the same features as training
    input_data = pd.DataFrame(
        0, index=[0], columns=FEATURE_COLUMNS
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

    if returned == "Yes":
        input_data["Returned_Yes"] = 1

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Profit: ₹{prediction:,.2f}")
