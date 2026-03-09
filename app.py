import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Resolve paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set page config
st.set_page_config(page_title="Ames Housing Dashboard", layout="wide", page_icon="🏡")
sns.set_theme(style="whitegrid", palette="viridis")

# --- LOAD DATA AND MODELS ---
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "AmesHousing.csv"))
    df_clean = df.drop(columns=['Order', 'PID'])
    X = df_clean.drop(columns=['SalePrice'])
    defaults = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            defaults[col] = X[col].mode()[0] if not X[col].dropna().empty else "None"
        else:
            defaults[col] = X[col].median() if not pd.isna(X[col].median()) else 0
    return df, X, defaults

@st.cache_resource
def load_models():
    models = {
        'Linear Regression': joblib.load(os.path.join(BASE_DIR, 'models/linear_regression.joblib')),
        'Decision Tree': joblib.load(os.path.join(BASE_DIR, 'models/decision_tree.joblib')),
        'Random Forest': joblib.load(os.path.join(BASE_DIR, 'models/random_forest.joblib')),
        'XGBoost': joblib.load(os.path.join(BASE_DIR, 'models/xgboost_model.joblib'))
    }
    mlp = joblib.load(os.path.join(BASE_DIR, 'models/mlp_model.joblib'))
    models['MLP'] = mlp

    preprocessor_linear = joblib.load(os.path.join(BASE_DIR, 'models/preprocessor_linear.joblib'))
    preprocessor_tree = joblib.load(os.path.join(BASE_DIR, 'models/preprocessor_tree.joblib'))
    shap_explainer = joblib.load(os.path.join(BASE_DIR, 'models/shap_explainer.joblib'))
    mlp_y_scaler = joblib.load(os.path.join(BASE_DIR, 'models/mlp_y_scaler.joblib'))

    with open(os.path.join(BASE_DIR, 'results/best_params.json'), 'r') as f:
        best_params = json.load(f)

    metrics = pd.read_csv(os.path.join(BASE_DIR, 'results/model_comparison.csv'))
    return models, preprocessor_linear, preprocessor_tree, shap_explainer, mlp_y_scaler, best_params, metrics

try:
    df, X_raw, default_vals = load_data()
    models_dict, prep_linear, prep_tree, explainer, mlp_y_scaler, params_dict, metrics_df = load_models()
except Exception as e:
    st.error(f"Error loading models or data. Ensure `analysis.py` has been executed to generate them. Details: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏡 Ames Housing Project")
    st.markdown("**MSIS 522 - Analytics and Machine Learning**")
    st.markdown("University of Washington, Foster School of Business")
    st.divider()
    st.markdown("### Top Metrics")
    st.metric("Best Model", "XGBoost", "Winner")
    st.metric("Highest R²", "0.916")
    st.metric("Lowest Test RMSE", "$24,348")
    st.divider()
    st.info("💡 **Navigation Hint:** Explore Tab 4 for an interactive home price calculator featuring real-time SHAP analysis.")

# --- MAIN DASHBOARD HEADER ---
st.title("🏡 Ames Housing Dataset — Regression Dashboard")
st.markdown("An end-to-end predictive analytics framework evaluating residential property values.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary", 
    "Descriptive Analytics", 
    "Model Performance", 
    "Explainability & Prediction"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tab1:
    # ROW OF METRIC CARDS
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Dataset Size", "2,930 Homes")
    col2.metric("Features", "80 Predictors")
    col3.metric("Best Model", "XGBoost")
    col4.metric("Best R² Score", "0.916")
    col5.metric("Best Test RMSE", "$24,348")
    st.divider()
    
    # HIGHLIGHT CALLOUT
    st.success("🏆 **Key Finding:** The highest ROI drivers of a home's value in this dataset are **Overall Quality**, **Above Ground Living Area (Sq. Ft.)**, and **Total Basement Square Footage**.")
    
    # SECTION: ABOUT THE DATASET
    st.markdown("### About the Dataset")
    st.markdown("""
    This project explores the Ames Housing dataset, which consists of 2,930 residential property sales recorded in Ames, Iowa. The primary target variable is the `SalePrice`, representing the final market value of the home in USD, ranging from approximately &#36;12,800 for teardown units up to &#36;755,000 for elite luxury estates. To predict this value, the dataset provides 80 predictor features spanning a wide range of property characteristics:
    - **Lot & Location:** lot size, zoning, neighborhood, proximity to roads/railroads
    - **Structure:** above-ground living area, basement square footage, number of bedrooms/bathrooms, building type
    - **Quality & Condition:** overall quality (1-10 scale), kitchen quality, exterior condition, functional rating
    - **Extras:** garage size and finish, fireplaces, pool, porch/deck area, fencing
    - **Sale Info:** month/year sold, sale type, sale condition
    
    By systematically analyzing these comprehensive attributes, we can isolate exactly which tangible, physical dimensions most heavily dictate a property's final free-market worth.
    """, unsafe_allow_html=True)
    
    # SECTION: WHY THIS MATTERS
    st.markdown("### Why This Matters")
    st.markdown("""
    Accurate home price prediction is a foundational pillar of the modern data-driven real estate economy. It directly helps prospective buyers avoid significantly overpaying by understanding realistic fair market values, while simultaneously helping sellers price their assets competitively so they sell faster. Furthermore, commercial lenders rely heavily on robust valuation models to securely assess mortgage risk during financial underwriting. Finally, understanding precisely which physical features computationally drive value empowers current homeowners to strategically prioritize renovations with the absolute highest return on investment, while neighborhood-level insights help investors identify uniquely undervalued geographic areas.
    """)
    
    # SECTION: APPROACH & KEY FINDINGS
    st.markdown("### Approach & Key Findings")
    st.markdown("""
    Our analytical workflow progressed fundamentally from initial Exploratory Data Analysis (EDA) and robust preprocessing into rigorous model training, evaluation, and SHAP explainability. We trained and evaluated five distinct machine learning models (Linear Regression, Decision Tree, Random Forest, XGBoost, and a deep MLP Neural Network) utilizing a strict randomized 70/30 train-test split, 5-fold cross-validation, and GridSearchCV for optimal hyperparameter tuning. Ultimately, the gradient-boosted tree architecture XGBoost emerged as the best performer with an **R² of 0.916** and **RMSE of approx. &#36;24,348**, cleanly outperforming the Linear Regression baseline (R² = 0.899) and the standard Decision Tree (R² = 0.864). Comprehensive SHAP extraction conclusively proved that the top three physical drivers of valuation are a home's Overall baseline Quality, its total Above-Ground Living Area, and its native Basement Square Footage.
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # SECTION B: KEY TAKEAWAYS
    st.markdown("### Key Takeaways")
    st.info("📊 **Data Insight:** Home prices in Ames range from &#36;12,800 to &#36;755,000 with a right-skewed distribution. The majority of homes are successfully sold tightly between &#36;100K and &#36;250K.")
    st.info("🏠 **Top Price Drivers:** Overall Quality, Above Ground Living Area, and Total Basement Square Footage are the three strongest independent predictors driving final sale prices globally across the market.")
    st.info("🏘️ **Neighborhood Matters:** Premium neighborhoods like Northridge Heights (`NridgHt`) and Northridge (`NoRidge`) command the highest median prices, easily doubling the valuations of entry-level locations like Meadow Village (`MeadowV`).")
    st.info("📉 **Model Performance:** Five competing methodologies were tested (Linear Regression, Decision Tree, Random Forest, XGBoost, and an MLP Neural Network). Tree-based ensemble estimators systematically extracted non-linear correlations superior to standard regression techniques.")
    st.info("🤖 **Winning Architecture:** XGBoost cleanly outperformed all other models, capturing over 91.5% of the total target price variance and securing the lowest error margins (approx. &#36;24,348 RMSE).")
    st.info("🔍 **Explainability:** Secondary SHAP analytics strictly map exactly how features interact. For example, moving a home's raw Overall Quality up by a single categorical tier increases the median predicted price by approximately &#36;24,500.")
    
    st.divider()
    
    # SECTION C: METHODOLOGY
    with st.expander("Methodology Details", expanded=False):
        st.markdown("""
        - **Data Partitioning:** Evaluated against a strict, randomized 70/30 train-test split (`random_state=42`) using cross-validation to prevent observational data leakage.
        - **Pipeline Preprocessing:** Skewed distributions were resolved using iterative median imputation, robust one-hot/ordinal categorical encodings, and numerical `StandardScaler` transformations.
        - **Algorithm Tuning:** Deep hyperparameter grids were dynamically evaluated iteratively through `GridSearchCV` featuring rigid 5-fold folds.
        - **Feature Interpretation:** Advanced Game-Theoretic `SHAP` values reverse-engineered the complex gradient boosting tree mechanics into simple, dollar-valued feature importance waterfall metrics.
        """)

# ==========================================
# TAB 2: DESCRIPTIVE ANALYTICS
# ==========================================
with tab2:
    st.header("Descriptive Analytics & EDA")
    
    st.subheader("Target Distribution")
    dist_img = os.path.join(BASE_DIR, "results/target_dist.png")
    if os.path.exists(dist_img):
        st.image(dist_img, use_column_width=True)
    st.markdown("**Insight:** The target variable `SalePrice` is strongly right-skewed, exhibiting a long tail for highly expensive properties exceeding the &#36;500,000 mark. The vast majority of standard homes are successfully sold between &#36;100,000 and &#36;250,000. Because of this pronounced right skewness, logarithmic transformation of the SalePrice is an effective strategy to help normalize the variance for linear models, reducing the proportional error on luxury outliers.", unsafe_allow_html=True)
    st.divider()
    
    st.subheader("Feature Relationships")
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(os.path.join(BASE_DIR, "results/overall_qual_boxplot.png")):
            st.image(os.path.join(BASE_DIR, "results/overall_qual_boxplot.png"), use_column_width=True)
        st.markdown("**Overall Quality vs Price:** There is a pronounced, exponential positive correlation between overall structural quality and sale price. Variance widens dramatically for the highest-tier homes (ratings 8-10) showing that luxury finishes drastically inflate market ceilings.")
        
        if os.path.exists(os.path.join(BASE_DIR, "results/neighborhood_bar.png")):
            st.image(os.path.join(BASE_DIR, "results/neighborhood_bar.png"), use_column_width=True)
        st.markdown("**Neighborhood Influence:** Location heavily dictates real estate valuations. Neighborhoods like 'NoRidge' and 'NridgHt' fetch premium median prices averaging well over &#36;300,000, while entry-level neighborhoods like 'MeadowV' or 'IDOTRR' hover tightly around &#36;100,000.", unsafe_allow_html=True)
        
        if os.path.exists(os.path.join(BASE_DIR, "results/year_built_scatter.png")):
            st.image(os.path.join(BASE_DIR, "results/year_built_scatter.png"), use_column_width=True)
        st.markdown("**Year Built vs Price:** Newer constructions reliably command premium valuations, demonstrating a steady positive linear trendline scaling continuously upward post-1950. However, fascinatingly dense clusters of highly-valued historic properties persist for homes built prior to 1920, reflecting the unique market demand surrounding historically preserved housing.")
        
        if os.path.exists(os.path.join(BASE_DIR, "results/garage_cars_box.png")):
            st.image(os.path.join(BASE_DIR, "results/garage_cars_box.png"), use_column_width=True)
        st.markdown("**Garage Capacity:** Property values jump significantly in the transitions strictly from zero, to one, and then optimally to a two-car garage format. The data reveals steeply diminishing returns passing the 3-car threshold, implying that standard multi-vehicle coverage saturates buyer demands.")
        
    with col2:
        if os.path.exists(os.path.join(BASE_DIR, "results/gr_liv_area_scatter.png")):
            st.image(os.path.join(BASE_DIR, "results/gr_liv_area_scatter.png"), use_column_width=True)
        st.markdown("**Living Area vs Price:** Above-ground living area scales almost linearly with price, acting as one of the most reliable prediction baselines. We observe a strict correlation though minor outliers exist where enormous footprints nevertheless sold cheaply, potentially due to remarkably poor condition constraints.")
        
        if os.path.exists(os.path.join(BASE_DIR, "results/bldg_type_box.png")):
            st.image(os.path.join(BASE_DIR, "results/bldg_type_box.png"), use_column_width=True)
        st.markdown("**Building Type:** Single-family homes represent the overwhelming bulk of the local market and inherently contain the highest variance in sales prices. Townhome End units also command very strong margins compared to classic two-family conversions, indicating high demand for attached luxury.")
        
        if os.path.exists(os.path.join(BASE_DIR, "results/central_air_box.png")):
            st.image(os.path.join(BASE_DIR, "results/central_air_box.png"), use_column_width=True)
        st.markdown("**Central Air Market Baseline:** The integration of modern central air conditioning operates almost categorically as a baseline gatekeeper for upscale property evaluation. Homes lacking central air systems suffer devastating price ceilings rarely escaping the &#36;150,000 limit, making it a critical fundamental investment.", unsafe_allow_html=True)
        
    st.divider()
    
    st.subheader("Neighborhood Categorical Intersections")
    if os.path.exists(os.path.join(BASE_DIR, "results/qual_neigh_bar.png")):
        st.image(os.path.join(BASE_DIR, "results/qual_neigh_bar.png"), use_column_width=True)
    st.markdown("**Premium Multiplier:** This grouped configuration tracking the top 5 elite neighborhoods proves that location exponentially scales base structural quality into massive profits. Noticeably, achieving an 'Overall Qual' tier of 9 or 10 inside premium districts uniquely propels average valuations dangerously close to &#36;500,000.", unsafe_allow_html=True)
    st.divider()
    
    st.subheader("Correlation Heatmap")
    if os.path.exists(os.path.join(BASE_DIR, "results/corr_heatmap.png")):
        st.image(os.path.join(BASE_DIR, "results/corr_heatmap.png"), use_column_width=True)
    st.markdown("""
    **Understanding the Correlations:** The heatmap above reveals that `Overall Qual` (0.80) and `Gr Liv Area` (0.71) represent the strongest independent linear drivers of a home's worth. Features associated with storage sizing also score highly, notably `Garage Cars` (0.65) and `Total Bsmt SF` (0.63). 
    
    However, the heatmap also highlights acute multicollinearity natively present in residential data. We can observe extreme feature correlations between `Garage Cars` and `Garage Area` (0.89), as well as `Total Bsmt SF` and `1st Flr SF` (0.80). This redundancy implies that while standard Linear Regression may suffer statistically from overlapping coefficients and inflated variance, our ensemble tree-based models (like Random Forest and XGBoost) are inherently immune to these multicollinearity faults and can seamlessly extract non-linear value from redundant parameters.
    """)

# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
with tab3:
    st.header("Predictive Model Comparison Overview")
    
    # 1. Model Comparison Summary
    st.markdown("### 1. Test Set Metrics Summary")
    if not metrics_df.empty:
        st.dataframe(metrics_df.set_index('Model').style.format("{:,.2f}"), use_container_width=True)
        
    st.markdown("""
    **Model Performance Trade-offs:** The validation results clearly indicate that **XGBoost performed best**, delivering an R² over 0.915 and securing the lowest RMSE by a considerable margin. Our **Decision Tree was the weakest performer**, suffering from clear overfitting and depth constraints ($RMSE \\approx &#36;30,900$). Surprisingly, the standard **Linear Regression** model established an exceptionally strong baseline performance ($R² = 0.898$), keeping tight pace with the complex Neural Network. The fundamental trade-off here lies between interpretation and accuracy—while Ridge Linear Regression explicitly details parameter coefficients linearly, the highly accurate XGBoost operates natively as a black box requiring secondary SHAP algorithms to explain identical decisions.
    """, unsafe_allow_html=True)
    if os.path.exists(os.path.join(BASE_DIR, "results/model_rmse_compare.png")):
        st.image(os.path.join(BASE_DIR, "results/model_rmse_compare.png"), use_column_width=True)
        
    st.divider()
    
    # 2. Individual Model Results
    st.header("Individual Model Deep Dives")
    
    with st.expander("Linear Regression (Ridge)", expanded=False):
        st.subheader("Linear Regression Performance")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(os.path.join(BASE_DIR, "results/lr_pred_actual.png")):
                st.image(os.path.join(BASE_DIR, "results/lr_pred_actual.png"), use_column_width=True)
            st.caption("Points plotting closer to the red diagonal indicate accurate predictions. The linear model holds well up to &#36;300K but struggles slightly on massive valuation properties.", unsafe_allow_html=True)
        with c2:
            if os.path.exists(os.path.join(BASE_DIR, "results/lr_residuals.png")):
                st.image(os.path.join(BASE_DIR, "results/lr_residuals.png"), use_column_width=True)
            st.caption("The residuals plot exhibits a subtle funneling (heteroscedasticity) effect toward higher price bounds, suggesting variances inflate alongside premium property tiers.")

    with st.expander("Decision Tree", expanded=False):
        st.subheader("Decision Tree Performance")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(os.path.join(BASE_DIR, "results/dt_pred_actual.png")):
                st.image(os.path.join(BASE_DIR, "results/dt_pred_actual.png"), use_column_width=True)
            st.caption("Notice the clear 'staircase' clustering pattern globally typical of un-ensembled tree architectures, forcing unique sale prices into rigid horizontal bucket bins.")
        with c2:
            st.empty()
            
        st.markdown("#### Truncated Tree Structure Visualization")
        if os.path.exists(os.path.join(BASE_DIR, "results/decision_tree_viz.png")):
            st.image(os.path.join(BASE_DIR, "results/decision_tree_viz.png"), use_column_width=True)

    with st.expander("Random Forest", expanded=False):
        st.subheader("Random Forest Performance")
        if os.path.exists(os.path.join(BASE_DIR, "results/rf_pred_actual.png")):
            st.image(os.path.join(BASE_DIR, "results/rf_pred_actual.png"), use_column_width=True)
        st.caption("While heavily performant, the Random Forest model's predictions occasionally begin to underestimate the severe variance of luxury properties crossing the &#36;500K threshold compared to boosted equivalents.", unsafe_allow_html=True)

    with st.expander("XGBoost (Best Model)", expanded=True):
        st.subheader("XGBoost Performance")
        if os.path.exists(os.path.join(BASE_DIR, "results/xgb_pred_actual.png")):
            st.image(os.path.join(BASE_DIR, "results/xgb_pred_actual.png"), use_column_width=True)
        st.caption("XGBoost demonstrates exceptionally tight clustering universally around the diagonal, succeeding flawlessly across mid-range home prices and managing high-end outliers effectively.")

    with st.expander("MLP Neural Network", expanded=False):
        st.subheader("Multi-Layer Perceptron (MLP)")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(os.path.join(BASE_DIR, "results/mlp_pred_actual.png")):
                st.image(os.path.join(BASE_DIR, "results/mlp_pred_actual.png"), use_column_width=True)
            st.caption("Scaling target variables allowed the Deep MLP architecture to correctly optimize and mirror Ridge-level outputs, keeping firm pace with ensemble frameworks.")
        with c2:
            if os.path.exists(os.path.join(BASE_DIR, "results/mlp_loss_curve.png")):
                st.image(os.path.join(BASE_DIR, "results/mlp_loss_curve.png"), use_column_width=True)
            st.caption("Cross-validation training curves demonstrate clean MSE descent convergence safely devoid of divergent overfitting spikes.")

    st.divider()
    
    # 3. Best Hyperparameters
    st.markdown("### Optimal Hyperparameters (GridSearchCV)")
    parsed_params = []
    for model_nm, p_dict in params_dict.items():
        for param_nm, p_val in p_dict.items():
            parsed_params.append({"Algorithm": model_nm, "Hyperparameter": param_nm, "Optimized Selection": p_val})
    st.dataframe(pd.DataFrame(parsed_params), use_container_width=True, hide_index=True)


# ==========================================
# TAB 4: EXPLAINABILITY & PREDICTION
# ==========================================
with tab4:
    st.header("Global Feature Explainability (SHAP)")
    
    st.markdown("""
    While tree-based models offer superior predictive accuracy over traditional linear approaches, they fundamentally lack transparent interpretability. We apply **SHAP (SHapley Additive exPlanations)** mathematics to peek deeply inside the XGBoost framework, explaining precisely which variables drove the logic tree matrices globally, and exclusively by how much.
    
    - **Strongest Predictors:** The absolute mean impact confirms that the top ranking features dictating final price are `Overall Qual` (shifting predictions by approx. &#36;24K universally on average), followed sequentially by `Gr Liv Area` (approx. &#36;12.5K), `Total Bsmt SF` (approx. &#36;6K), and `Garage Cars` (approx. &#36;4.5K). Note that features suffering from extreme scarcity or severe sparsity (such as `Pool QC`) may occasionally trigger highly localized mathematical artifacts inside isolated subsets, but their true global impact scale remains strictly minimal.
    - **Directional Influence:** The beeswarm scatterplot clearly highlights that high recorded feature values (colored red) in key attributes precisely like `Overall Qual` and `Gr Liv Area` aggressively propel the predicted price positively to the right. Conversely, poor overall ratings and highly strained physical sizes reliably deduct cleanly from the expected baseline average.
    - **Decisive Utility for Stakeholders:** These insights offer incredibly concrete, highly actionable directives. An established homeowner considering intensive renewals to directly maximize market listing ROI should consistently focus almost exclusively on macro-quality overall upgrades (modernizing integrated systems/finishes to rapidly boost `Overall Qual`) and fully refining their basement layout potential, rather than wasting capital adding disproportionate minor features. A prospective buyer conversely should ruthlessly prioritize newer, wholly structurally sound premium homes firmly anchored inside elite preferred core neighborhoods.
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Feature Influence (Beeswarm Plot)")
        if os.path.exists(os.path.join(BASE_DIR, "results/shap_summary.png")):
            st.image(os.path.join(BASE_DIR, "results/shap_summary.png"), use_column_width=True)
    with col2:
        st.markdown("#### Mean Absolute Impact (Bar Plot)")
        if os.path.exists(os.path.join(BASE_DIR, "results/shap_bar.png")):
            st.image(os.path.join(BASE_DIR, "results/shap_bar.png"), use_column_width=True)
            
    st.divider()
    
    st.markdown("#### Static Trace: Explaining the Most Expensive Dataset Home")
    if os.path.exists(os.path.join(BASE_DIR, "results/shap_waterfall_static.png")):
        st.image(os.path.join(BASE_DIR, "results/shap_waterfall_static.png"), use_column_width=True)
        st.caption("This static waterfall diagram definitively traces how the native XGBoost model mathematically validated the expected price precisely for the single highest-valued property in the strict training subset. Launching dynamically from the underlying market base average price established at approx. &#36;181,000, compounding variables explicitly including uniquely outstanding `Overall Qual`, sprawling unconstrained `Gr Liv Area`, and premium exterior finishes overwhelmingly successfully drive the highly calculated terminal valuation smoothly past &#36;600,000 vertically.", unsafe_allow_html=True)
        
    st.divider()
    
    st.header("Interactive Real-Time Prediction")
    st.markdown("Configure diverse feature bounds below and observe immediately dynamically how exactly the resulting final property price fluctuates leveraging our heavily pre-trained pipeline.")
    
    st.subheader("Model Configuration & House Details")
    # First row: Model and fundamental categorical details
    c1, c2, c3 = st.columns(3)
    with c1:
        model_choice = st.selectbox("Select Target ML Architecture", list(models_dict.keys()), index=list(models_dict.keys()).index('XGBoost') if 'XGBoost' in models_dict else 0)
    with c2:
        neighborhood = st.selectbox("Designated Neighborhood", sorted(df['Neighborhood'].unique().tolist()), index=list(sorted(df['Neighborhood'].unique())).index(default_vals.get('Neighborhood', 'NAmes')) if 'Neighborhood' in df else 0)
    with c3:
        year_built = st.slider("Absolute Year Constructed", 1870, 2010, int(default_vals.get('Year Built', 1970)))
        
    st.subheader("Quality & Size Metrics")
    # Second row: High impact dimensional & quality sliders grouped together
    s1, s2 = st.columns(2)
    with s1:
        overall_qual = st.slider("Overall Structural Quality Rank (1-10)", 1, 10, int(default_vals.get('Overall Qual', 6)))
        gr_liv_area = st.number_input("Above Ground Living Space (Sq. Ft.)", 500, 6000, int(default_vals.get('Gr Liv Area', 1500)), step=100)
    with s2:
        total_bsmt_sf = st.number_input("Foundational Total Basement Area (Sq. Ft.)", 0, 4000, int(default_vals.get('Total Bsmt SF', 1000)), step=100)
        garage_cars = st.slider("Integrated Garage Capacity", 0, 5, int(default_vals.get('Garage Cars', 2)))
        full_bath = st.slider("Confirmed Full Bathrooms", 0, 4, int(default_vals.get('Full Bath', 2)))
        
    st.divider()
    
    # Prediction logic
    user_data = default_vals.copy()
    user_data['Overall Qual'] = overall_qual
    user_data['Gr Liv Area'] = gr_liv_area
    user_data['Neighborhood'] = neighborhood
    user_data['Year Built'] = year_built
    user_data['Total Bsmt SF'] = total_bsmt_sf
    user_data['Garage Cars'] = garage_cars
    user_data['Full Bath'] = full_bath
    
    user_df = pd.DataFrame([user_data])
    
    try:
        if model_choice == 'Linear Regression' or model_choice == 'MLP':
            X_proc = prep_linear.transform(user_df)
        else:
            X_proc = prep_tree.transform(user_df)
            
        active_model = models_dict[model_choice]
        prediction = active_model.predict(X_proc)
        
        # Flatten and inverse scale MLP prediction array
        if model_choice == 'MLP':
            prediction = prediction.flatten()
            prediction = mlp_y_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
            
        # Large Metric Output
        mc1, mc2, mc3 = st.columns([1, 2, 1])
        with mc2:
            st.metric(label=f"Predicted Local Sale Price ({model_choice})", value=f"${prediction[0]:,.2f}", delta="Estimated Final Valuation")
        
        # Waterfall Plot for Tree models
        if model_choice in ['XGBoost', 'Random Forest', 'Decision Tree']:
            st.markdown(f"#### Custom Individual Trajectory Trace ({model_choice})")
            st.markdown("This live isolated waterfall plot directly mathematically documents exactly how precisely your individually configured active slider parameters mathematically stacked dynamically to tangibly shift positively or negatively sharply from the base standard housing baseline evaluation constraints.")
            
            # Recreate feature names
            num_cols = df.drop(columns=['SalePrice', 'Order', 'PID']).select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = df.drop(columns=['SalePrice', 'Order', 'PID']).select_dtypes(include=['object']).columns.tolist()
            tree_feature_names = num_cols + cat_cols
            X_proc_df = pd.DataFrame(X_proc, columns=tree_feature_names)
            
            shap_values_single = explainer(X_proc_df)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values_single[0], show=False)
            st.pyplot(fig)
        else:
            st.info("💡 Advanced contextual SHAP waterfall tree logic visualization exclusively inherently demands deep ensembled tree-based frameworks precisely for mapping. Instantly switch back the configured model selector cleanly to `XGBoost`, `Random Forest`, or `Decision Tree` natively above to mathematically isolate and completely review absolute custom single-prediction SHAP breakdowns.")
            
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
