import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('expanded_lenskart_site_feasibility_500.csv')

# Drop unnecessary columns
columns_to_drop = ['site_id', 'address', 'landmark', 'zip_code', 'property_type', 
                   'site_area_sqft', 'asking_rent_per_sqft_inr', 'floor_level']
df = df.drop(columns_to_drop, axis=1)

# Replace 9999.0 with NaN in numerical columns
numerical_cols = ['latitude', 'longitude', 'num_optical_stores_within_1km', 'num_eye_clinics_within_1km',
                  'distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km',
                  'distance_to_public_transport_km', 'num_retail_shops_within_1km',
                  'num_restaurants_within_1km', 'estimated_daily_foot_traffic',
                  'vehicular_traffic_count']
df[numerical_cols] = df[numerical_cols].replace(9999.0, np.nan)

# Define categorical columns
categorical_cols = ['city', 'state', 'locality_type']

# Preprocessing for numerical data: impute missing values with median, then standardize
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: one-hot encode
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Split the data into features and target
X = df.drop('feasibility_score', axis=1)
y = df['feasibility_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')

# Extract and display feature importances
feature_names = (numerical_cols + 
                 pipeline.named_steps['preprocessor']
                 .transformers_[1][1]
                 .get_feature_names_out(categorical_cols).tolist())
importances = pipeline.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nFeature Importances:")
print(feature_importance_df.sort_values('Importance', ascending=False).to_string(index=False))