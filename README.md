# Flight-price-and-customer-satisfaction-prediction
**Overview**
This project consists of two interconnected machine learning applications aimed at enhancing the travel and tourism domain:
    1) Flight Price Prediction (Regression)
    2) Customer Satisfaction Prediction (Classification) 
Both applications are integrated into a single Streamlit app with separate pages for each project. Additionally, MLflow is used for model tracking and management.
# Project 1: Flight Price Prediction (Regression)
**Objective**
Develop an end-to-end machine learning pipeline to predict flight ticket prices based on factors such as departure time, source, destination, and airline type. Deploy the pipeline in an interactive Streamlit app for real-time predictions.

**Skills Utilized**
• Python • Machine Learning • Data Analysis • Streamlit • MLflow

**Methodology**
1. Data Preprocessing
• Handle missing values and duplicates. • Convert date/time columns into standard formats. • Engineer new features (e.g., Duration_Hours, Duration_Minutes).

2. Model Development
• Perform exploratory data analysis (EDA) to identify trends and correlations. • Train regression models (Linear Regression, Random Forest, XGBoost, etc). • Log experiments, parameters, and metrics using MLflow.

3. Streamlit App Development
• Visualize trends in flight prices. • Enable user input for route, airline, and date/time to predict prices.

# Project 2: Customer Satisfaction Prediction (Classification)
**Objective**
Develop a classification model to predict customer satisfaction levels based on feedback, demographics, and service ratings. Integrate the model into a Streamlit app for interactive prediction.

**Skills Utilized**
• Python • Machine Learning • Data Analysis • Streamlit • MLflow

**Methodology**
1. Data Preprocessing
• Handle missing values and duplicates. • Encode categorical features. • Normalize numerical features.

2. Model Development
• Perform EDA to understand feature relationships. • Train classification models (Logistic Regression, Random Forest, Gradient Boosting, etc.,). • Log experiments, metrics (accuracy, F1-score), and confusion matrices using MLflow.

3. Streamlit App Development
• Visualize customer satisfaction trends. • Allow user input for features to predict satisfaction levels.

# Streamlit App Structure
**Overview**
The Streamlit app integrates both projects into two distinct pages:
    1) Flight Price Prediction
    2) Customer Satisfaction Prediction
**Features**
• Page Navigation: Sidebar navigation to switch between projects. • Interactive Inputs: Users can input features such as route, airline, time, and service ratings. • Visualizations: Display trends and feature importance. • Real-time Predictions: Provide results based on user inputs.

**Technical Implementation**
• Regression Page: Implements the flight price prediction pipeline. • Classification Page: Implements the customer satisfaction prediction pipeline. • Shared Components: Visualization tools and MLflow integration for tracking experiments.

**Tools and Technologies**
• Python Libraries: pandas, numpy, sklearn, xgboost, streamlit, matplotlib, seaborn • Deployment: Streamlit • Model Tracking: MLflow

**Conclusion**
This project combines machine learning and data analysis to predict flight prices and customer satisfaction levels, addressing key challenges in the travel and tourism industry. The user-friendly Streamlit app provides accurate predictions and actionable insights, backed by robust preprocessing and advanced models. With MLflow integration for model tracking, this solution demonstrates the power of data-driven decision-making to enhance customer experiences and optimize travel planning.
