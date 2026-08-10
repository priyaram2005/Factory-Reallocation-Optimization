# 🏭 Factory Reallocation & Shipping Optimization Recommendation System

## 📌 Project Overview

This project was developed as part of the Unified Mentor Data Science Internship.

The objective is to build an intelligent decision support system for Nassau Candy Distributor that recommends optimal factory reallocations to improve shipping efficiency while maintaining profitability.

The system predicts shipping lead time using Machine Learning and simulates multiple factory assignment scenarios before recommending the best option.

---

## 🎯 Problem Statement

Nassau Candy Distributor currently assigns products to factories using static business rules.

This results in:

- Higher shipping distances
- Longer delivery lead times
- Increased logistics cost
- Reduced operational efficiency

This project provides an intelligent recommendation engine capable of simulating alternative factory assignments before implementation.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Feature Engineering
- Shipping Distance Calculation
- Predictive Machine Learning Models
- Factory Reallocation Simulation
- What-If Scenario Analysis
- Recommendation Engine
- Risk & Impact Analysis
- Interactive Streamlit Dashboard

---

## 🧠 Machine Learning Models

The following regression models were trained and evaluated:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor ✅ (Best Model)

Evaluation Metrics:

- RMSE
- MAE
- R² Score

---

## 📊 Dashboard Modules

### Executive Dashboard

- KPI Cards
- Sales Analysis
- Profit Analysis
- Factory Utilization
- Shipping Distance Distribution

### Factory Optimization Simulator

- Select Product
- Select Factory
- Select Region
- Select Ship Mode
- Predict Lead Time

### What-If Analysis

- Compare Current vs Alternative Factory
- Lead Time Comparison
- Distance Comparison

### Recommendation Dashboard

- Top Factory Recommendations
- Optimization Score
- Download Recommendations
- Interactive Charts

### Risk & Impact Panel

- High Risk Recommendations
- Optimization Score Distribution
- Business Impact Summary

---

## 📂 Project Structure

```
Factory-Reallocation-Optimization/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── Nassau Candy Distributor.csv
│   ├── prepared_data.csv
│   └── factory_recommendations.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── artifacts.pkl
│
└── notebooks/
    └── model_training.ipynb
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

---

## 📈 Key Performance Indicators

- Lead Time Reduction
- Shipping Distance
- Profit Margin
- Optimization Score
- Recommendation Coverage

---

## ▶️ How to Run

### Clone the repository

```bash
git clone <repository-url>
```

### Navigate to the project

```bash
cd Factory-Reallocation-Optimization
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📷 Dashboard

The Streamlit dashboard includes:

- Executive Dashboard
- Factory Optimization Simulator
- What-If Analysis
- Recommendation Dashboard
- Risk & Impact Panel

---

## 👩‍💻 Author

**Priya**

Data Science Intern

Unified Mentor Pvt. Ltd.

---

## 📄 License

This project was developed for educational and internship purposes.