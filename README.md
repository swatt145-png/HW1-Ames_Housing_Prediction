# Ames Housing Dataset — Regression Task

## Overview
This project is an end-to-end data science application predicting `SalePrice` for the Ames Housing dataset. It covers exploratory data analysis (EDA), predictive modeling with multiple algorithms (Linear Regression, Decision Tree, Random Forest, XGBoost, and an MLP Neural Network), explainability using SHAP, and an interactive Streamlit web application.

## Project Structure
- `app.py`: The main Streamlit application containing the UI and dashboard.
- `analysis.ipynb`: A Jupyter Notebook containing all EDA, modeling, and SHAP logic, structured according to the project requirements.
- `requirements.txt`: Python package dependencies.
- `models/`: Directory where trained models and pipelines are saved.
- `results/`: Directory where output metrics, charts, and configuration are persisted.

## How to Run Locally

1. **Install Requirements:**
   Make sure you have Python 3.9+ installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Models & Results (Optional, models are pre-trained):**
   Open and execute `analysis.ipynb` from start to finish via Jupyter:
   ```bash
   jupyter notebook analysis.ipynb
   ```
   This will run EDA, train all models, compute SHAP explainers, and save the resulting objects to the `models/` and `results/` folders respectively.

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   Access the app at `http://localhost:8501`.

## Tabs in the Streamlit App
- **Tab 1: Executive Summary** - Non-technical business report describing dataset, approach, and key findings.
- **Tab 2: Descriptive Analytics** - Pre-computed charts demonstrating distributions and associations.
- **Tab 3: Model Performance** - Comparison of multiple predictive models.
- **Tab 4: Explainability & Interactive Prediction** - Learn how features influence predictions globally, and compute prices dynamically using the stored models.
