# %% [markdown]
# # Part 1: Descriptive Analytics
# 
# ## 1.1 Dataset Introduction
# The Ames Housing dataset contains 2,930 residential property sales in Ames, Iowa, with 82 columns detailing physical characteristics of the homes (such as lot size, above-ground living area, number of rooms, garage/basement details) as well as condition/quality ratings, neighborhood information, and sale specifics.
# 
# The target variable for our predictive modeling is **SalePrice** (continuous, in USD). 
# 
# **Why this is interesting:** Predicting home prices is crucial for multiple stakeholders. It helps buyers to avoid overpaying, sellers to set competitive listing prices, and real estate professionals to advise their clients effectively. Understanding precisely which features (e.g., overall quality, square footage, neighborhood) most significantly drive a home's value can also empower homeowners to make cost-effective renovation and home improvement decisions with the highest ROI.
# 
# **Basic Statistics:** The raw dataset comprises 2,930 rows and 82 columns (39 numerical and 43 categorical features).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
import shap
import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# Set global professional theme
sns.set_theme(style="whitegrid", palette="viridis")
dollar_format = ticker.StrMethodFormatter('${x:,.0f}')

df = pd.read_csv("AmesHousing.csv")
print(f"Dataset shape: {df.shape}")
print(f"SalePrice Range: ${df['SalePrice'].min():,.0f} to ${df['SalePrice'].max():,.0f}")
print(f"SalePrice Mean: ${df['SalePrice'].mean():,.0f}")

# %% [markdown]
# ## 1.2 Target Distribution
# The plot below shows the distribution of our target variable, `SalePrice`. The distribution is clearly **right-skewed** with a long tail of very expensive properties. Most homes sell between $100K and $250K, but there are notable outliers above $500K. Because of this right skewness, a log transformation of the SalePrice may be beneficial for linear models to help normalize the variance, though tree-based models generally handle it well.

# %%
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, bins=50, color='#2a9d8f', ax=ax)
ax.xaxis.set_major_formatter(dollar_format)
plt.title('Distribution of SalePrice', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Sale Price ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('results/target_dist.png')
plt.close()

# %% [markdown]
# ## 1.3 Feature Distributions and Relationships
# Exploring how different features relate to the `SalePrice`.

# %%
# 1. Boxplot of SalePrice by Overall Quality
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Overall Qual', y='SalePrice', data=df, palette='viridis', ax=ax)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Sale Price by Overall Quality', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Overall Quality Rating (1-10)', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/overall_qual_boxplot.png')
plt.close()

# %%
# 2. Scatter plot of Gr Liv Area vs. SalePrice
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df, alpha=0.6, color='#264653', ax=ax)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Sale Price vs. Above Ground Living Area', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Above Ground Living Area (sq ft)', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/gr_liv_area_scatter.png')
plt.close()

# %%
# 3. Bar chart of median SalePrice by Neighborhood
neighborhood_medians = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=neighborhood_medians.index, y=neighborhood_medians.values, palette='mako', ax=ax)
ax.yaxis.set_major_formatter(dollar_format)
plt.xticks(rotation=90)
plt.title('Median Sale Price by Neighborhood', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Neighborhood', fontsize=12)
plt.ylabel('Median Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/neighborhood_bar.png')
plt.close()

# %%
# 4. Boxplot of SalePrice by Building Type
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Bldg Type', y='SalePrice', data=df, palette='Set2', ax=ax)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Sale Price by Building Type', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Building Type', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/bldg_type_box.png')
plt.close()

# %%
# 5. Scatter plot of Year Built vs. SalePrice with trend line
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='Year Built', y='SalePrice', data=df, scatter_kws={'alpha':0.4, 'color':'#2a9d8f'}, line_kws={'color':'#e76f51'}, ax=ax)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Sale Price vs. Year Built', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Year Built', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/year_built_scatter.png')
plt.close()

# %%
# 6. Boxplot of SalePrice by Central Air
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Central Air', y='SalePrice', data=df, palette='Set2', ax=ax)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Sale Price by Central Air', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Central Air (Y/N)', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/central_air_box.png')
plt.close()

# %%
# 7. Grouped bar chart of SalePrice by Overall Qual and Neighborhood (Top 5)
top_5_neigh = neighborhood_medians.head(5).index.tolist()
df_top5 = df[df['Neighborhood'].isin(top_5_neigh)]
fig, ax = plt.subplots(figsize=(14, 7))
sns.barplot(x='Neighborhood', y='SalePrice', hue='Overall Qual', data=df_top5, palette='viridis', errorbar=None, ax=ax)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Sale Price by Quality across Top 5 Neighborhoods', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Neighborhood', fontsize=12)
plt.ylabel('Average Sale Price ($)', fontsize=12)
plt.legend(title='Overall Qual', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/qual_neigh_bar.png')
plt.close()

# %%
# 8. Boxplot of Garage Cars vs SalePrice
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Garage Cars', y='SalePrice', data=df, palette='viridis', ax=ax)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Sale Price by Garage Car Capacity', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Garage Cars capacity', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/garage_cars_box.png')
plt.close()

# %% [markdown]
# ## 1.4 Correlation Heatmap

# %%
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sale_price_corr = corr_matrix[['SalePrice']].sort_values(by='SalePrice', ascending=False)
top_features = sale_price_corr.head(15).index

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix.loc[top_features, top_features], annot=True, cmap='mako', fmt=".2f", square=True)
plt.title('Top 15 Correlated Features with SalePrice', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('results/corr_heatmap.png')
plt.close()

# %% [markdown]
# # Part 2: Predictive Analytics
# 
# ## 2.1 Data Preparation

# %%
# Drop identifiers
df_clean = df.drop(columns=['Order', 'PID'])

# Define Features and Target
X = df_clean.drop(columns=['SalePrice'])
y = df_clean['SalePrice']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Identify numerical and categorical columns
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Create Preprocessing Pipelines
num_pipeline_linear = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

num_pipeline_tree = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

cat_pipeline_linear = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

cat_pipeline_tree = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor_linear = ColumnTransformer([
    ('num', num_pipeline_linear, num_cols),
    ('cat', cat_pipeline_linear, cat_cols)
])

preprocessor_tree = ColumnTransformer([
    ('num', num_pipeline_tree, num_cols),
    ('cat', cat_pipeline_tree, cat_cols)
])

X_train_linear = preprocessor_linear.fit_transform(X_train)
X_test_linear = preprocessor_linear.transform(X_test)
joblib.dump(preprocessor_linear, 'models/preprocessor_linear.joblib')

X_train_tree = preprocessor_tree.fit_transform(X_train)
X_test_tree = preprocessor_tree.transform(X_test)
joblib.dump(preprocessor_tree, 'models/preprocessor_tree.joblib')

tree_feature_names = num_cols + cat_cols

results = []
best_params_dict = {}

def log_result(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results.append({'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    print(f"--- {model_name} ---")
    print(f"MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R2: {r2:.4f}\n")

# %% [markdown]
# ## 2.2 Linear Regression Baseline

# %%
lr = Ridge(alpha=10.0, random_state=42)
lr.fit(X_train_linear, y_train)
y_pred_lr = lr.predict(X_test_linear)
log_result('Linear Regression (Ridge)', y_test, y_pred_lr)
joblib.dump(lr, 'models/linear_regression.joblib')

fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.5, color='#2a9d8f', ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.xaxis.set_major_formatter(dollar_format)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Linear Regression: Predicted vs Actual', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Sale Price ($)', fontsize=12)
plt.ylabel('Predicted Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/lr_pred_actual.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_pred_lr, y=y_test - y_pred_lr, alpha=0.5, color='#e76f51', ax=ax)
ax.axhline(0, color='r', linestyle='--', lw=2)
ax.xaxis.set_major_formatter(dollar_format)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Linear Regression: Residuals Plot', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Sale Price ($)', fontsize=12)
plt.ylabel('Residual Error ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/lr_residuals.png')
plt.close()

# %% [markdown]
# ## 2.3 Decision Tree / CART

# %%
dt = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20, 50]
}
gs_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs_dt.fit(X_train_tree, y_train)

best_dt = gs_dt.best_estimator_
best_params_dict['Decision Tree'] = gs_dt.best_params_
y_pred_dt = best_dt.predict(X_test_tree)
log_result('Decision Tree', y_test, y_pred_dt)
joblib.dump(best_dt, 'models/decision_tree.joblib')

# DT Predicted vs Actual
fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_dt, alpha=0.5, color='#264653', ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.xaxis.set_major_formatter(dollar_format)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Decision Tree: Predicted vs Actual', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Sale Price ($)', fontsize=12)
plt.ylabel('Predicted Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/dt_pred_actual.png')
plt.close()

# Truncated Decision Tree Viz (Fix overlapping + sizing)
plt.figure(figsize=(28, 14))
plot_tree(best_dt, feature_names=tree_feature_names, filled=True, rounded=True, max_depth=3, fontsize=12)
plt.title("Decision Tree Visualization (Max Depth = 3)", fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/decision_tree_viz.png')
plt.close()

# %% [markdown]
# ## 2.4 Random Forest

# %%
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 8]
}
gs_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs_rf.fit(X_train_tree, y_train)

best_rf = gs_rf.best_estimator_
best_params_dict['Random Forest'] = gs_rf.best_params_
y_pred_rf = best_rf.predict(X_test_tree)
log_result('Random Forest', y_test, y_pred_rf)
joblib.dump(best_rf, 'models/random_forest.joblib')

fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5, color='#2a9d8f', ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.xaxis.set_major_formatter(dollar_format)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('Random Forest: Predicted vs Actual', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Sale Price ($)', fontsize=12)
plt.ylabel('Predicted Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/rf_pred_actual.png')
plt.close()

# %% [markdown]
# ## 2.5 Boosted Trees — XGBoost

# %%
xgb_model = xgb.XGBRegressor(random_state=42)
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1]
}
gs_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs_xgb.fit(X_train_tree, y_train)

best_xgb = gs_xgb.best_estimator_
best_params_dict['XGBoost'] = gs_xgb.best_params_
y_pred_xgb = best_xgb.predict(X_test_tree)
log_result('XGBoost', y_test, y_pred_xgb)
joblib.dump(best_xgb, 'models/xgboost_model.joblib')

fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_xgb, alpha=0.5, color='#e76f51', ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.xaxis.set_major_formatter(dollar_format)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('XGBoost: Predicted vs Actual', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Sale Price ($)', fontsize=12)
plt.ylabel('Predicted Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/xgb_pred_actual.png')
plt.close()

# %% [markdown]
# ## 2.6 Neural Network — MLP
# Scaling the target variable (SalePrice) for MLP to avoid massive loss values (1e10 issues) and enable proper Adam optimization.

# %%
input_dim = X_train_linear.shape[1]

# Scale target for MLP
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
joblib.dump(y_scaler, 'models/mlp_y_scaler.joblib')

def build_mlp():
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='mse')
    return model

mlp = build_mlp()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = mlp.fit(
    X_train_linear, y_train_scaled,
    validation_split=0.2,
    epochs=150,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=0
)

# Plot scaled training/validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='#264653', lw=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='#e9c46a', lw=2)
plt.title('MLP Training & Validation Loss (Scaled Target)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE (Scaled)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('results/mlp_loss_curve.png')
plt.close()

# Evaluate on inverse-transformed predictions
y_pred_mlp_scaled = mlp.predict(X_test_linear).flatten()
y_pred_mlp = y_scaler.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).flatten()
log_result('MLP Neural Network', y_test, y_pred_mlp)
mlp.save('models/mlp_model.keras')

fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_mlp, alpha=0.5, color='#e9c46a', edgecolor='black', ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.xaxis.set_major_formatter(dollar_format)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('MLP Neural Network: Predicted vs Actual', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Sale Price ($)', fontsize=12)
plt.ylabel('Predicted Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('results/mlp_pred_actual.png')
plt.close()

# %% [markdown]
# ## 2.7 Model Comparison Summary

# %%
res_df = pd.DataFrame(results).set_index('Model')
res_df.to_csv('results/model_comparison.csv')

with open('results/best_params.json', 'w') as f:
    json.dump(best_params_dict, f, indent=4)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=res_df.index, y=res_df['RMSE'], palette='mako', ax=ax)
ax.yaxis.set_major_formatter(dollar_format)
plt.title('RMSE by Model Verification', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Root Mean Squared Error ($)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/model_rmse_compare.png')
plt.close()

# %% [markdown]
# # Part 3: Explainability — SHAP Analysis

# %%
X_train_tree_df = pd.DataFrame(X_train_tree, columns=tree_feature_names)

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer(X_train_tree_df)
joblib.dump(explainer, 'models/shap_explainer.joblib')

# Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_train_tree_df, max_display=10, show=False)
plt.savefig('results/shap_summary.png', bbox_inches='tight')
plt.close()

# Bar Plot
plt.figure()
shap.plots.bar(shap_values, max_display=10, show=False)
plt.savefig('results/shap_bar.png', bbox_inches='tight')
plt.close()

# Static Waterfall Plot for a specific interesting case (Most expensive home)
min_idx = y_train.argmin() 
max_idx = y_train.argmax()

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[max_idx], show=False)
plt.title("SHAP Waterfall: Most Expensive House in Train Set", fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig('results/shap_waterfall_static.png', bbox_inches='tight')
plt.close()

print("Execution fully complete.")
