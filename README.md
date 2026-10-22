# Sales Profit Prediction

## Project Overview
This project uses Machine Learning to predict sales profit in Indian Rupees (INR) based on order details, discounts, location, product category, shipping information, and other features.

An XGBoost regression model is used to estimate the expected profit for a sales transaction.

## Objectives
- Predict sales profit using machine learning.
- Explore sales transaction data.
- Compare regression models.
- Build an interactive web application using Streamlit.

## Dataset
- **Total records:** 15,000
- **Total original columns:** 21
- **Target variable:** `Profit_INR`
- **Problem type:** Regression

The dataset contains order dates, customer and product information, regions, sales channels, discounts, shipping details, and profit information.

## Machine Learning Models
The following models were evaluated:

| Model | Mean 5-Fold CV R² |
|---|---:|
| XGBoost | 81.50% |
| Gradient Boosting | 81.30% |
| Random Forest | 80.91% |
| Decision Tree | 65.72% |

**Selected model:** XGBoost Regressor

XGBoost achieved a test R² score of approximately **0.8190**.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## Project Structure

```text
Sales-Profit-Prediction/
├── app.py
├── sales_data_model.pkl
├── requirements.txt
└── README.md
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

2. Navigate to the project folder:

```bash
cd YOUR_REPOSITORY
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Enter the order details and click **Predict Profit** to view the estimated profit in INR.

## Model Evaluation

The model was evaluated using:
- R² score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- 5-fold cross-validation

The XGBoost model achieved a test R² score of approximately 0.8190, indicating that it explains about 81.9% of the variation in the test-set profit values.

## Future Improvements
- Tune XGBoost hyperparameters.
- Improve feature engineering.
- Add visualizations for sales and profit analysis.
- Deploy the application online.
- Validate predictions using only information available at the intended prediction time.

## Author
**Farooq Khan**

Aspiring Machine Learning Engineer

## Disclaimer
Predictions are estimates based on the training data and may not reflect actual business profit.
