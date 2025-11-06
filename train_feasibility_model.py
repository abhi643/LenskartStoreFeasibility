import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- Load the dataset ---
df = pd.read_csv('expanded_lenskart_site_feasibility_500.csv')

# --- Drop unnecessary columns ---
columns_to_drop = [
    'site_id', 'address', 'landmark', 'zip_code', 'property_type',
    'site_area_sqft', 'asking_rent_per_sqft_inr', 'floor_level'
]
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], axis=1)

# --- Define placeholder values to normalize ---
PLACEHOLDERS = { -999, -999.0, 9999, 9999.0 }

# --- Identify feature groups ---
numerical_cols = [
    'latitude', 'longitude', 'num_optical_stores_within_1km', 'num_eye_clinics_within_1km',
    'distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km',
    'distance_to_public_transport_km', 'num_retail_shops_within_1km',
    'num_restaurants_within_1km', 'estimated_daily_foot_traffic',
    'vehicular_traffic_count'
]

categorical_cols = ['city', 'state', 'locality_type']

# --- Clean numeric placeholders ---
for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].replace(list(PLACEHOLDERS), np.nan)

# --- Normalize categorical columns ---
for col in categorical_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "None": np.nan})
        )

# --- Split into features and target ---
target = 'feasibility_score'
X = df.drop(columns=[target])
y = df[target]

# --- Build preprocessing pipeline ---
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, [c for c in numerical_cols if c in X.columns]),
    ('cat', categorical_transformer, [c for c in categorical_cols if c in X.columns])
])

# --- Define model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)

# --- Combine into final pipeline ---
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
pipeline.fit(X_train, y_train)

# --- Save model ---
joblib.dump(pipeline, 'feasibility_model.joblib')
print("✅ Model training complete and saved to feasibility_model.joblib")

# --- Save schema for validation during inference ---
feature_schema = {
    "numerical": [c for c in numerical_cols if c in X.columns],
    "categorical": [c for c in categorical_cols if c in X.columns],
    "all_features_order": X.columns.tolist(),
    "target": target
}

with open("feasibility_model.schema.json", "w") as f:
    json.dump(feature_schema, f, indent=2)

# --- Evaluate model ---
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"📊 Mean Squared Error: {mse:.4f}")
print(f"📈 R-squared: {r2:.4f}")

# --- Feature importances ---
feature_names = (
    [c for c in numerical_cols if c in X.columns]
    + pipeline.named_steps['preprocessor']
    .transformers_[1][1]
    .get_feature_names_out([c for c in categorical_cols if c in X.columns]).tolist()
)
importances = pipeline.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

print("\n🏆 Feature Importances:")
print(feature_importance_df.sort_values('Importance', ascending=False).to_string(index=False))