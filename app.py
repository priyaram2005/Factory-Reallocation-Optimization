import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------

st.set_page_config(
    page_title="Factory Reallocation & Shipping Optimization",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:12px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.08);
    text-align:center;
}

h1,h2,h3{
    color:#003366;
}

.sidebar .sidebar-content{
    background-color:#002B5B;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_DIR / "prepared_data.csv")

    rec = pd.read_csv(DATA_DIR / "factory_recommendations.csv")

    return df, rec

df, recommendation_df = load_data()

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

@st.cache_resource
def load_models():

    model = joblib.load(MODEL_DIR / "best_model.pkl")

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")

    return model, scaler, label_encoders

model, scaler, label_encoders = load_models()

# -------------------------------------------------------
# KPI VALUES
# -------------------------------------------------------

total_orders = len(df)

total_sales = df["Sales"].sum()

gross_profit = df["Gross Profit"].sum()

avg_lead = df["Lead Time"].mean()

avg_distance = df["Shipping Distance (km)"].mean()

total_products = df["Product Name"].nunique()

total_factories = df["Factory"].nunique()

coverage = recommendation_df["Product"].nunique()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.image(
    "https://img.icons8.com/fluency/96/factory.png",
    width=90
)

st.sidebar.title("Nassau Candy")

st.sidebar.markdown("### Factory Optimization System")

page = st.sidebar.radio(

    "Navigation",

    [

        "Executive Dashboard",

        "Factory Optimization",

        "What-If Analysis",

        "Recommendation Dashboard",

        "Risk & Impact",

        "About Project"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info(

"""
Machine Learning Model

Gradient Boosting Regressor

Project:
Factory Reallocation &
Shipping Optimization

Unified Mentor
"""
)

# -------------------------------------------------------
# EXECUTIVE DASHBOARD
# -------------------------------------------------------

if page == "Executive Dashboard":

    st.title("🏭 Factory Reallocation & Shipping Optimization")

    st.markdown(
        "### Executive Business Dashboard"
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    c2.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

    c3.metric(
        "Gross Profit",
        f"${gross_profit:,.2f}"
    )

    c4.metric(
        "Products",
        total_products
    )

    c5,c6,c7,c8 = st.columns(4)

    c5.metric(
        "Average Lead Time",
        f"{avg_lead:.2f}"
    )

    c6.metric(
        "Average Distance (km)",
        f"{avg_distance:.2f}"
    )

    c7.metric(
        "Factories",
        total_factories
    )

    c8.metric(
        "Recommendation Coverage",
        coverage
    )

    st.markdown("---")

    left,right = st.columns(2)

    with left:

        sales_region = (

            df.groupby("Region")["Sales"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            sales_region,

            x="Region",

            y="Sales",

            title="Sales by Region",

            color="Sales",

            text_auto=".2s"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with right:

        profit_division = (

            df.groupby("Division")["Gross Profit"]

            .sum()

            .reset_index()

        )

        fig = px.pie(

            profit_division,

            names="Division",

            values="Gross Profit",

            hole=0.45,

            title="Profit Contribution by Division"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    left,right = st.columns(2)

    with left:

        factory_count = (

            df["Factory"]

            .value_counts()

            .reset_index()

        )

        factory_count.columns = [

            "Factory",

            "Orders"

        ]

        fig = px.bar(

            factory_count,

            x="Factory",

            y="Orders",

            color="Orders",

            title="Factory Utilization"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with right:

        fig = px.histogram(

            df,

            x="Shipping Distance (km)",

            nbins=35,

            title="Shipping Distance Distribution"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.markdown("---")

    st.subheader("Top Selling Products")

    top_products = (

        df.groupby("Product Name")["Sales"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

        .reset_index()

    )

    fig = px.bar(

        top_products,

        x="Sales",

        y="Product Name",

        orientation="h",

        color="Sales",

        text_auto=".2s"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

# ==========================================================
# LOAD ARTIFACTS
# ==========================================================

artifacts = joblib.load(MODEL_DIR / "artifacts.pkl")

feature_columns = artifacts["feature_columns"]

categorical_columns = artifacts["categorical_columns"]

numeric_columns = artifacts["numeric_columns"]

factory_coordinates = artifacts["factory_coordinates"]

# ==========================================================
# HAVERSINE DISTANCE FUNCTION
# ==========================================================

from math import radians, sin, cos, sqrt, atan2


def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_lead_time(sample):

    temp = pd.DataFrame([sample])

    temp = temp[feature_columns]

    for col in categorical_columns:

        temp[col] = label_encoders[col].transform(
            temp[col].astype(str)
        )

    temp[numeric_columns] = scaler.transform(
        temp[numeric_columns]
    )

    prediction = model.predict(temp)[0]

    return prediction


# ==========================================================
# FACTORY OPTIMIZATION PAGE
# ==========================================================

if page == "Factory Optimization":

    st.title("🏭 Factory Optimization Simulator")

    st.write(
        "Predict shipping lead time for any product by simulating different factory assignments."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        selected_product = st.selectbox(

            "Select Product",

            sorted(df["Product Name"].unique())

        )

        selected_factory = st.selectbox(

            "Select Factory",

            sorted(factory_coordinates.keys())

        )

    with col2:

        selected_region = st.selectbox(

            "Destination Region",

            sorted(df["Region"].unique())

        )

        selected_shipmode = st.selectbox(

            "Ship Mode",

            sorted(df["Ship Mode"].unique())

        )

    sample = (

        df[df["Product Name"] == selected_product]

        .iloc[0]

        .copy()

    )

    sample["Factory"] = selected_factory
    sample["Region"] = selected_region
    sample["Ship Mode"] = selected_shipmode

    sample["Factory Latitude"] = factory_coordinates[selected_factory][0]
    sample["Factory Longitude"] = factory_coordinates[selected_factory][1]

    distance = haversine(

        sample["Factory Latitude"],

        sample["Factory Longitude"],

        sample["Customer Latitude"],

        sample["Customer Longitude"]

    )

    sample["Shipping Distance (km)"] = distance

    predicted = predict_lead_time(sample)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "Predicted Lead Time",

        f"{predicted:.2f}"

    )

    c2.metric(

        "Shipping Distance",

        f"{distance:.2f} km"

    )

    c3.metric(

        "Profit Margin",

        f"{sample['Profit Margin']:.2%}"

    )

    fig = go.Figure()

    fig.add_trace(

        go.Indicator(

            mode="gauge+number",

            value=predicted,

            title={"text": "Predicted Lead Time"},

            gauge={

                "axis": {"range": [0, max(df["Lead Time"])]}

            }

        )

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


# ==========================================================
# WHAT IF ANALYSIS
# ==========================================================

if page == "What-If Analysis":

    st.title("🔄 What-If Scenario Analysis")

    st.write(
        "Compare the current factory against an alternative factory."
    )

    st.markdown("---")

    product = st.selectbox(

        "Product",

        sorted(df["Product Name"].unique())

    )

    current = (

        df[df["Product Name"] == product]

        .iloc[0]

        .copy()

    )

    current_factory = current["Factory"]

    alternative_factory = st.selectbox(

        "Alternative Factory",

        sorted(factory_coordinates.keys())

    )

    current_prediction = predict_lead_time(current)

    alternative = current.copy()

    alternative["Factory"] = alternative_factory

    alternative["Factory Latitude"] = factory_coordinates[alternative_factory][0]

    alternative["Factory Longitude"] = factory_coordinates[alternative_factory][1]

    new_distance = haversine(

        alternative["Factory Latitude"],

        alternative["Factory Longitude"],

        alternative["Customer Latitude"],

        alternative["Customer Longitude"]

    )

    alternative["Shipping Distance (km)"] = new_distance

    new_prediction = predict_lead_time(alternative)

    left, right = st.columns(2)

    with left:

        st.subheader("Current Factory")

        st.metric(

            "Factory",

            current_factory

        )

        st.metric(

            "Lead Time",

            f"{current_prediction:.2f}"

        )

        st.metric(

            "Distance",

            f"{current['Shipping Distance (km)']:.2f} km"

        )

    with right:

        st.subheader("Alternative Factory")

        st.metric(

            "Factory",

            alternative_factory

        )

        st.metric(

            "Lead Time",

            f"{new_prediction:.2f}"

        )

        st.metric(

            "Distance",

            f"{new_distance:.2f} km"

        )

    comparison = pd.DataFrame({

        "Scenario": [

            "Current",

            "Alternative"

        ],

        "Lead Time": [

            current_prediction,

            new_prediction

        ]

    })

    fig = px.bar(

        comparison,

        x="Scenario",

        y="Lead Time",

        color="Scenario",

        text_auto=".2f",

        title="Lead Time Comparison"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    improvement = current_prediction - new_prediction

    if improvement > 0:

        st.success(

            f"Estimated Lead Time Improvement: {improvement:.2f} days"

        )

    elif improvement < 0:

        st.error(

            f"Lead Time Increases by {abs(improvement):.2f} days"

        )

    else:

        st.info(

            "No difference in predicted lead time."

        )

# ==========================================================
# RECOMMENDATION DASHBOARD
# ==========================================================

if page == "Recommendation Dashboard":

    st.title("💡 Factory Reallocation Recommendations")

    st.write(
        "Top recommended factory assignments generated using the Machine Learning optimization engine."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        product_filter = st.selectbox(
            "Filter by Product",
            ["All"] + sorted(recommendation_df["Product"].unique().tolist())
        )

    with col2:

        min_score = st.slider(
            "Minimum Optimization Score",
            float(recommendation_df["Optimization Score"].min()),
            float(recommendation_df["Optimization Score"].max()),
            float(recommendation_df["Optimization Score"].min())
        )

    filtered = recommendation_df.copy()

    if product_filter != "All":
        filtered = filtered[
            filtered["Product"] == product_filter
        ]

    filtered = filtered[
        filtered["Optimization Score"] >= min_score
    ]

    st.dataframe(
        filtered,
        use_container_width=True,
        height=500
    )

    st.download_button(

        label="📥 Download Recommendations",

        data=filtered.to_csv(index=False),

        file_name="factory_recommendations.csv",

        mime="text/csv"

    )

    st.markdown("---")

    top10 = (

        filtered

        .sort_values(

            "Optimization Score",

            ascending=False

        )

        .head(10)

    )

    fig = px.bar(

        top10,

        x="Optimization Score",

        y="Product",

        color="Recommended Factory",

        orientation="h",

        title="Top Recommended Factory Reallocations",

        text_auto=".2f"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    filtered = filtered.copy()

    filtered["Bubble Size"] = (
        filtered["Optimization Score"].abs() + 1
    )

    fig = px.scatter(
        filtered,
        x="Shipping Distance (km)",
        y="Predicted Lead Time",
        color="Recommended Factory",
        size="Bubble Size",
        hover_data=[
            "Product",
            "Optimization Score"
        ],
        title="Shipping Distance vs Predicted Lead Time"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# RISK & IMPACT PANEL
# ==========================================================

if page == "Risk & Impact":

    st.title("⚠ Risk & Impact Analysis")

    st.markdown("---")

    high_risk = recommendation_df[
        recommendation_df["Optimization Score"] < recommendation_df["Optimization Score"].median()
    ]

    st.subheader("High Risk Reassignments")

    st.dataframe(

        high_risk,

        use_container_width=True

    )

    st.markdown("---")

    fig = px.box(

        recommendation_df,

        y="Predicted Lead Time",

        color="Recommended Factory",

        title="Predicted Lead Time Distribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    fig = px.histogram(

        recommendation_df,

        x="Optimization Score",

        nbins=25,

        title="Optimization Score Distribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Business Impact Summary")

    st.success(

        f"""

Average Predicted Lead Time :

{recommendation_df['Predicted Lead Time'].mean():.2f}

Average Shipping Distance :

{recommendation_df['Shipping Distance (km)'].mean():.2f} km

Average Optimization Score :

{recommendation_df['Optimization Score'].mean():.2f}

"""
    )


# ==========================================================
# ABOUT PROJECT
# ==========================================================

if page == "About Project":

    st.title("About Project")

    st.markdown("---")

    st.markdown("""

# Factory Reallocation & Shipping Optimization Recommendation System

### Objective

This project helps Nassau Candy Distributor optimize product allocation across manufacturing factories.

The system predicts shipping lead times using Machine Learning and recommends better factory assignments to improve operational efficiency while maintaining profitability.

---

## Machine Learning Models

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor ✅ Best Model

---

## Features Used

- Product
- Factory
- Region
- Ship Mode
- Shipping Distance
- Sales
- Units
- Cost
- Gross Profit
- Profit Margin

---

## Dashboard Modules

- Executive Dashboard
- Factory Optimization Simulator
- What-If Scenario Analysis
- Recommendation Dashboard
- Risk & Impact Panel

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Streamlit

---

## Business Benefits

✔ Reduced Shipping Lead Time

✔ Factory Reallocation Simulation

✔ Shipping Distance Optimization

✔ Data-Driven Decision Support

✔ Profitability Analysis

""")

    st.markdown("---")

    st.info(

        """
Unified Mentor Internship Project

Factory Reallocation & Shipping Optimization Recommendation System

Prepared by:

Priya
"""
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center; color:gray;'>

Factory Reallocation & Shipping Optimization Recommendation System

Developed using Streamlit • Plotly • Scikit-Learn

© 2026

</div>
""",
unsafe_allow_html=True
)