# 🚗 Ford Car Price Prediction

A Machine Learning web application that predicts the price of a Ford car based on its specifications.
## 🖥️ Application Screenshot

![Ford Car Price Prediction](screenshot.png.png)
## 📌 Project Overview

This project uses Machine Learning to predict the price of a Ford car using:

- Car Model
- Manufacturing Year
- Transmission
- Mileage
- Fuel Type
- Tax
- MPG
- Engine Size

The trained Linear Regression model is deployed as an interactive Streamlit web application.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- GitHub

## 🤖 Machine Learning Model

**Algorithm:** Linear Regression

### Preprocessing

Categorical features are handled using One-Hot Encoding.

The preprocessing and model are combined using a Scikit-learn Pipeline.

## 📊 Input Features

| Feature | Description |
|---|---|
| Model | Ford car model |
| Year | Manufacturing year |
| Transmission | Transmission type |
| Mileage | Distance travelled |
| Fuel Type | Type of fuel |
| Tax | Vehicle tax |
| MPG | Miles per gallon |
| Engine Size | Engine capacity |

## 📈 Model Evaluation

## 📈 Model Evaluation

The Linear Regression model was evaluated on the test dataset using R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

| Metric | Score |
|---|---:|
| R² Score | 0.8388 |
| MAE | 1,398.29 |
| RMSE | 1,900.92 |

### Evaluation Metrics

- **R² Score:** 0.8388 — the model explains approximately 83.88% of the variation in car prices.
- **MAE:** 1,398.29 — the average absolute difference between predicted and actual prices.
- **RMSE:** 1,900.92 — the square root of the average squared prediction error.

## 🚀 Run Locally

### Clone the repository

```bash
git clone https://github.com/PragyaSaini05/ford-car-price-prediction.git
