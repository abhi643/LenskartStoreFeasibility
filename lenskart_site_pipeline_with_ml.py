# import requests
# import uuid
# import time
# import pandas as pd
# import numpy as np
# import logging
# from geopy.distance import geodesic
# import csv
# import joblib
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import os
# from ratelimit import limits, sleep_and_retry

# # ------------- Config -------------
# OVERPASS_URL = "http://overpass-api.de/api/interpreter"
# NOMINATIM_URL = "https://nominatim.openstreetmap.org"
# USER_AGENT = {"User-Agent": "lenskart-site-feasibility-pipeline/2.0"}
# ML_DEFAULT_NUMERIC = 9999  # Default value for missing numeric fields
# OUTPUT_CSV = "expanded_lenskart_site_feasibility_500.csv"  # Configurable output path
# ONE_SECOND = 1  # Rate limit period for API calls

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # ------------- Rate-limited API Call -------------
# @sleep_and_retry
# @limits(calls=1, period=ONE_SECOND)
# def make_api_request(url, params=None, data=None, headers=None):
#     try:
#         response = requests.get(url, params=params, headers=headers) if params else requests.post(url, data=data, headers=headers)
#         response.raise_for_status()
#         return response
#     except requests.exceptions.RequestException as e:
#         logger.warning(f"API request failed: {e}, Status Code: {response.status_code if 'response' in locals() else 'N/A'}, Response: {response.text if 'response' in locals() else 'N/A'}")
#         raise

# # ------------- Train and Save Model -------------
# def train_and_save_model(data_file, model_file='feasibility_model.joblib', save_preprocessed=True, preprocessed_file='preprocessed_data_for_visualization.csv'):
#     logger.info("Training the model...")
#     try:
#         # Load the dataset
#         df = pd.read_csv(data_file)
#     except FileNotFoundError:
#         logger.error(f"Input file not found: {data_file}")
#         raise
#     except PermissionError:
#         logger.error(f"Permission denied when reading file: {data_file}")
#         raise
    
#     # Store original data for reference
#     original_df = df.copy()
    
#     # Drop unnecessary columns
#     columns_to_drop = ['site_id', 'address', 'landmark', 'zip_code', 'property_type', 
#                        'site_area_sqft', 'asking_rent_per_sqft_inr', 'floor_level']
#     df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
    
#     # Replace 9999.0 with NaN in numerical columns
#     numerical_cols = ['latitude', 'longitude', 'num_optical_stores_within_1km', 'num_eye_clinics_within_1km',
#                       'distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km',
#                       'distance_to_public_transport_km', 'num_retail_shops_within_1km',
#                       'num_restaurants_within_1km', 'estimated_daily_foot_traffic',
#                       'vehicular_traffic_count']
#     # Validate numerical columns
#     missing_cols = [col for col in numerical_cols if col not in df.columns]
#     if missing_cols:
#         logger.error(f"Missing expected columns: {missing_cols}")
#         raise ValueError(f"Missing expected columns: {missing_cols}")
    
#     df[numerical_cols] = df[numerical_cols].replace(9999.0, np.nan)
    
#     # Define categorical columns
#     categorical_cols = ['city', 'state', 'locality_type']
    
#     # Save preprocessed data for visualization if requested
#     if save_preprocessed:
#         try:
#             # Create a copy of the preprocessed data for visualization
#             viz_df = df.copy()
            
#             # Add back some useful columns from original data for visualization context
#             viz_df['site_id'] = original_df['site_id']
#             viz_df['address'] = original_df['address']
#             viz_df['landmark'] = original_df['landmark']
#             viz_df['zip_code'] = original_df['zip_code']
#             viz_df['property_type'] = original_df['property_type']
#             viz_df['site_area_sqft'] = original_df['site_area_sqft']
#             viz_df['asking_rent_per_sqft_inr'] = original_df['asking_rent_per_sqft_inr']
#             viz_df['floor_level'] = original_df['floor_level']
            
#             # Reorder columns to have identifiers first
#             id_cols = ['site_id', 'address', 'landmark', 'zip_code', 'property_type', 
#                       'site_area_sqft', 'asking_rent_per_sqft_inr', 'floor_level']
#             other_cols = [col for col in viz_df.columns if col not in id_cols]
#             viz_df = viz_df[id_cols + other_cols]
            
#             # Save to CSV
#             viz_df.to_csv(preprocessed_file, index=False)
#             logger.info(f"Preprocessed data saved to {preprocessed_file} for visualization purposes")
            
#         except Exception as e:
#             logger.warning(f"Failed to save preprocessed data: {e}")
    
#     # Preprocessing for numerical data
#     numerical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler())
#     ])
    
#     # Preprocessing for categorical data
#     categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
#     # Combine preprocessing steps
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_transformer, numerical_cols),
#             ('cat', categorical_transformer, categorical_cols)
#         ])
    
#     # Define the model
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
    
#     # Create the pipeline
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('model', model)
#     ])
    
#     # Split the data
#     X = df.drop('feasibility_score', axis=1)
#     y = df['feasibility_score']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train the model
#     pipeline.fit(X_train, y_train)
    
#     # Evaluate the model
#     y_pred = pipeline.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     logger.info(f'Model trained - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}')
    
#     # Save the model
#     try:
#         joblib.dump(pipeline, model_file)
#         logger.info(f"Model saved to {model_file}")
#     except PermissionError:
#         logger.error(f"Permission denied when saving model to {model_file}")
#         raise
#     return pipeline

# # ------------- Preprocess Data for Visualization -------------
# def preprocess_data_for_visualization(input_file, output_file='preprocessed_data_for_visualization.csv'):
#     logger.info(f"Preprocessing {input_file} for visualization...")
    
#     try:
#         # Load the dataset
#         df = pd.read_csv(input_file)
#         logger.info(f"Loaded {len(df)} rows from {input_file}")
#     except FileNotFoundError:
#         logger.error(f"Input file not found: {input_file}")
#         raise
#     except PermissionError:
#         logger.error(f"Permission denied when reading file: {input_file}")
#         raise
        
#     # Create a copy for preprocessing
#     processed_df = df.copy()
    
#     # Define numerical columns that need preprocessing
#     numerical_cols = ['latitude', 'longitude', 'num_optical_stores_within_1km', 'num_eye_clinics_within_1km',
#                       'distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km',
#                       'distance_to_public_transport_km', 'num_retail_shops_within_1km',
#                       'num_restaurants_within_1km', 'estimated_daily_foot_traffic',
#                       'vehicular_traffic_count']
    
#     # Validate numerical columns
#     missing_cols = [col for col in numerical_cols if col not in processed_df.columns]
#     if missing_cols:
#         logger.error(f"Missing expected columns: {missing_cols}")
#         raise ValueError(f"Missing expected columns: {missing_cols}")
    
#     # Replace 9999.0 with NaN in numerical columns
#     for col in numerical_cols:
#         processed_df[col] = processed_df[col].replace(9999.0, np.nan)
#         logger.info(f"Processed column {col}: {processed_df[col].isnull().sum()} missing values")
    
#     # Add preprocessing indicators for visualization
#     processed_df['has_missing_coordinates'] = processed_df['latitude'].isnull() | processed_df['longitude'].isnull()
#     processed_df['has_missing_poi_data'] = processed_df[['num_optical_stores_within_1km', 'num_eye_clinics_within_1km']].isnull().any(axis=1)
#     processed_df['has_missing_distance_data'] = processed_df[['distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km', 'distance_to_public_transport_km']].isnull().any(axis=1)
#     processed_df['has_missing_traffic_data'] = processed_df[['estimated_daily_foot_traffic', 'vehicular_traffic_count']].isnull().any(axis=1)
    
#     # Add data quality score
#     total_important_cols = len(numerical_cols)
#     processed_df['data_quality_score'] = processed_df[numerical_cols].notna().sum(axis=1) / total_important_cols if total_important_cols > 0 else 0
    
#     # Add feasibility score categories
#     if 'feasibility_score' in processed_df.columns:
#         processed_df['feasibility_category'] = pd.cut(
#             processed_df['feasibility_score'], 
#             bins=[0, 3, 5, 7, 10], 
#             labels=['Low', 'Medium', 'High', 'Very High'],
#             include_lowest=True
#         )
    
#     # Add region grouping
#     if 'state' in processed_df.columns:
#         north_states = ['Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Himachal Pradesh', 'Uttarakhand']
#         south_states = ['Karnataka', 'Tamil Nadu', 'Andhra Pradesh', 'Telangana', 'Kerala']
#         west_states = ['Maharashtra', 'Gujarat', 'Goa']
#         east_states = ['West Bengal', 'Odisha', 'Jharkhand', 'Bihar']
#         central_states = ['Madhya Pradesh', 'Chhattisgarh', 'Rajasthan']
        
#         def get_region(state):
#             if state in north_states:
#                 return 'North'
#             elif state in south_states:
#                 return 'South'
#             elif state in west_states:
#                 return 'West'
#             elif state in east_states:
#                 return 'East'
#             elif state in central_states:
#                 return 'Central'
#             else:
#                 return 'Other'
        
#         processed_df['region'] = processed_df['state'].apply(get_region)
    
#     # Save the preprocessed data
#     try:
#         processed_df.to_csv(output_file, index=False)
#         logger.info(f"Preprocessed data saved to {output_file}")
#         logger.info(f"Preprocessed data shape: {processed_df.shape}")
#     except PermissionError:
#         logger.error(f"Permission denied when saving file: {output_file}")
#         raise
    
#     # Print summary statistics
#     print("\n=== Preprocessing Summary ===")
#     print(f"Total records: {len(processed_df)}")
#     print(f"Records with missing coordinates: {processed_df['has_missing_coordinates'].sum()}")
#     print(f"Records with missing POI data: {processed_df['has_missing_poi_data'].sum()}")
#     print(f"Records with missing distance data: {processed_df['has_missing_distance_data'].sum()}")
#     print(f"Records with missing traffic data: {processed_df['has_missing_traffic_data'].sum()}")
#     print(f"Average data quality score: {processed_df['data_quality_score'].mean():.2f}")
    
#     if 'feasibility_category' in processed_df.columns:
#         print("\nFeasibility Score Distribution:")
#         print(processed_df['feasibility_category'].value_counts())
    
#     if 'region' in processed_df.columns:
#         print("\nRegional Distribution:")
#         print(processed_df['region'].value_counts())
    
#     return processed_df

# # ------------- Geocoding Functions -------------
# def sleep():
#     time.sleep(1.2)

# def geocode_address(address):
#     logger.info(f"Geocoding address: {address}")
#     address_parts = [part.strip() for part in address.split(',')]
#     cleaned_parts = [part for part in address_parts if not any(kw in part.lower() for kw in ['shop no', 'unit no', 'near'])]
#     cleaned_address = ', '.join(cleaned_parts)
    
#     street, city, state, postcode = '', '', '', ''
#     for part in cleaned_parts:
#         part_lower = part.lower()
#         if any(kw in part_lower for kw in ['road', 'street', 'nagar', 'complex', 'plot']):
#             street = part if not street else f"{street}, {part}"
#         elif any(kw in part_lower for kw in ['indore', 'mumbai', 'delhi', 'bangalore', 'kolkata', 'chennai']):
#             city = part
#         elif any(kw in part_lower for kw in ['madhya pradesh', 'maharashtra', 'karnataka', 'west bengal', 'tamil nadu']):
#             state = part
#         elif part.replace(' ', '').isdigit():
#             postcode = part
    
#     params = {k: v for k, v in {'street': street, 'city': city, 'state': state, 'postalcode': postcode, 'country': 'India', 'format': 'json', 'limit': 1}.items() if v}
#     try:
#         r = make_api_request(f"{NOMINATIM_URL}/search", params=params, headers=USER_AGENT).json()
#         sleep()
#         if r and isinstance(r, list) and len(r) > 0 and 'lat' in r[0] and 'lon' in r[0]:
#             return float(r[0]['lat']), float(r[0]['lon'])
#     except Exception as e:
#         logger.warning(f"Structured query failed: {e}")
    
#     params = {'q': cleaned_address, 'format': 'json', 'limit': 1}
#     try:
#         r = make_api_request(f"{NOMINATIM_URL}/search", params=params, headers=USER_AGENT).json()
#         sleep()
#         if r and isinstance(r, list) and len(r) > 0 and 'lat' in r[0] and 'lon' in r[0]:
#             return float(r[0]['lat']), float(r[0]['lon'])
#     except Exception as e:
#         logger.warning(f"First fallback failed: {e}")
    
#     if city and state:
#         minimal_address = f"{city}, {state}"
#         logger.warning(f"Falling back to minimal address: {minimal_address}")
#         params = {'q': minimal_address, 'format': 'json', 'limit': 1}
#         try:
#             r = make_api_request(f"{NOMINATIM_URL}/search", params=params, headers=USER_AGENT).json()
#             sleep()
#             if r and isinstance(r, list) and len(r) > 0 and 'lat' in r[0] and 'lon' in r[0]:
#                 logger.warning(f"Using approximate coordinates for {minimal_address}")
#                 return float(r[0]['lat']), float(r[0]['lon'])
#         except Exception as e:
#             logger.warning(f"Second fallback failed: {e}")
    
#     logger.error(f"Geocoding failed for {address}")
#     return None, None

# def reverse_geocode(lat, lon):
#     params = {'lat': lat, 'lon': lon, 'format': 'json'}
#     try:
#         r = make_api_request(f"{NOMINATIM_URL}/reverse", params=params, headers=USER_AGENT).json()
#         sleep()
#         addr = r.get('address', {})
#         return (r.get('display_name', ''), addr.get('city', '') or addr.get('town', '') or addr.get('village', ''), addr.get('state', ''), addr.get('postcode', ''))
#     except Exception as e:
#         logger.error(f"Reverse geocoding failed: {e}")
#         return '', '', '', ''

# def get_locality_type(lat, lon):
#     params = {'lat': lat, 'lon': lon, 'format': 'json'}
#     try:
#         r = make_api_request(f"{NOMINATIM_URL}/reverse", params=params, headers=USER_AGENT).json()
#         sleep()
#         t = r.get('address', {}).get('city', '') or r.get('address', {}).get('town', '') or r.get('address', {}).get('village', '')
#         return "Urban" if t else "Rural" if r.get('address', {}).get('hamlet', '') else "Semi-Urban"
#     except Exception as e:
#         logger.error(f"Locality type fetch failed: {e}")
#         return "Semi-Urban"

# def get_better_landmark(lat, lon):
#     priorities = [('railway_station', 'amenity'), ('subway_entrance', 'railway'), ('mall', 'shop'), ('hospital', 'amenity'), ('school', 'amenity'), ('bank', 'amenity'), ('park', 'leisure'), ('hotel', 'tourism'), ('marketplace', 'amenity'), ('restaurant', 'amenity'), ('parking', 'amenity'), ('bus_station', 'amenity'), ('cinema', 'amenity'), ('college', 'amenity'), ('university', 'amenity')]
#     for key, tag_type in priorities:
#         query = f"[out:json][timeout:25];(node(around:400,{lat},{lon})[{tag_type}=\"{key}\"][name];way(around:400,{lat},{lon})[{tag_type}=\"{key}\"][name];rel(around:400,{lat},{lon})[{tag_type}=\"{key}\"][name];);out center 1;"
#         try:
#             r = make_api_request(OVERPASS_URL, data={'data': query}, headers=USER_AGENT).json()
#             sleep()
#             if r.get('elements', []):
#                 name = r['elements'][0].get('tags', {}).get('name')
#                 if name:
#                     return name
#         except Exception as e:
#             logger.warning(f"Landmark query failed for {key}: {e}")
#     return "No landmark found"

# def count_pois(lat, lon, osm_query):
#     query = f"[out:json][timeout:25];(node{osm_query}(around:1000,{lat},{lon});way{osm_query}(around:1000,{lat},{lon});rel{osm_query}(around:1000,{lat},{lon}););out count;"
#     try:
#         r = make_api_request(OVERPASS_URL, data={'data': query}, headers=USER_AGENT).json()
#         sleep()
#         return r.get('elements', [{}])[0].get('tags', {}).get('total', 0)
#     except Exception as e:
#         logger.warning(f"POI count failed for {osm_query}: {e}")
#         return ML_DEFAULT_NUMERIC

# def nearest_poi_distance(lat, lon, osm_query):
#     query = f"[out:json][timeout:25];(node{osm_query}(around:5000,{lat},{lon});way{osm_query}(around:5000,{lat},{lon});rel{osm_query}(around:5000,{lat},{lon}););out center;"
#     try:
#         r = make_api_request(OVERPASS_URL, data={'data': query}, headers=USER_AGENT).json()
#         sleep()
#         min_dist = ML_DEFAULT_NUMERIC
#         for el in r.get('elements', []):
#             poi_lat = el.get('lat') or el.get('center', {}).get('lat')
#             poi_lon = el.get('lon') or el.get('center', {}).get('lon')
#             if poi_lat and poi_lon:
#                 d = geodesic((lat, lon), (poi_lat, poi_lon)).km
#                 min_dist = min(min_dist, d) if min_dist != ML_DEFAULT_NUMERIC else d
#         return round(min_dist, 3) if min_dist != ML_DEFAULT_NUMERIC else ML_DEFAULT_NUMERIC
#     except Exception as e:
#         logger.warning(f"Nearest POI distance failed for {osm_query}: {e}")
#         return ML_DEFAULT_NUMERIC

# def estimate_foot_traffic(num_retail, num_restaurants, num_transport):
#     try:
#         num_retail = int(float(num_retail)) if num_retail is not None and str(num_retail).replace('.', '').replace('-', '').isdigit() else 0
#         num_restaurants = int(float(num_restaurants)) if num_restaurants is not None and str(num_restaurants).replace('.', '').replace('-', '').isdigit() else 0
#         num_transport = int(float(num_transport)) if num_transport is not None and str(num_transport).replace('.', '').replace('-', '').isdigit() else 0
#         return num_retail * 2 + num_restaurants * 2 + num_transport * 5
#     except (ValueError, TypeError):
#         logger.warning("Invalid input for foot traffic estimation, returning default value")
#         return ML_DEFAULT_NUMERIC

# def estimate_vehicular_traffic(lat, lon):
#     query = f"[out:json][timeout:25];(way(around:200,{lat},{lon})[highway~\"primary|secondary|trunk\"];);out;"
#     try:
#         r = make_api_request(OVERPASS_URL, data={'data': query}, headers=USER_AGENT).json()
#         sleep()
#         return len(r.get('elements', [])) * 50 if r.get('elements') else ML_DEFAULT_NUMERIC
#     except Exception as e:
#         logger.warning(f"Vehicular traffic query failed: {e}")
#         return ML_DEFAULT_NUMERIC

# # ------------- Main Pipeline Function with ML Prediction -------------
# def lenskart_site_pipeline_with_feasibility(address=None, lat=None, lon=None, property_type=None, site_area_sqft=None, asking_rent_per_sqft_inr=None, floor_level=None, model=None, output_csv=OUTPUT_CSV):
#     site_id = str(uuid.uuid4())
    
#     # Handle property data with defaults and validation
#     property_type = property_type.strip() if property_type and isinstance(property_type, str) and property_type.strip() else "commercial"
#     site_area_sqft = int(float(site_area_sqft)) if site_area_sqft and str(site_area_sqft).replace('.', '').replace('-', '').isdigit() and float(site_area_sqft) > 0 else ML_DEFAULT_NUMERIC
#     asking_rent_per_sqft_inr = int(float(asking_rent_per_sqft_inr)) if asking_rent_per_sqft_inr and str(asking_rent_per_sqft_inr).replace('.', '').replace('-', '').isdigit() and float(asking_rent_per_sqft_inr) > 0 else ML_DEFAULT_NUMERIC
#     floor_level = floor_level.strip() if floor_level and isinstance(floor_level, str) and floor_level.strip() else "Ground"
    
#     # Geocode if needed
#     if address and (lat is None or lon is None):
#         lat, lon = geocode_address(address)
#         if lat is None or lon is None:
#             logger.error(f"Geocoding failed for {address}")
#             return [site_id, None, None, address, "", "", "", "", "Geocoding Failed", ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, property_type, site_area_sqft, asking_rent_per_sqft_inr, floor_level, 0.0]
#     elif lat is None or lon is None:
#         logger.error("No address or lat/lon provided")
#         return [site_id, None, None, "", "", "", "", "", "No Location Provided", ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, ML_DEFAULT_NUMERIC, property_type, site_area_sqft, asking_rent_per_sqft_inr, floor_level, 0.0]

#     # Get location data
#     address_full, city, state, zip_code = reverse_geocode(lat, lon)
#     locality_type = get_locality_type(lat, lon)
#     landmark = get_better_landmark(lat, lon)
    
#     # Get POI data
#     num_optical_stores = count_pois(lat, lon, '["shop"="optician"]')
#     num_eye_clinics = count_pois(lat, lon, '["healthcare"="clinic"]["healthcare:speciality"="optometry"]')
#     dist_nearest_optical = nearest_poi_distance(lat, lon, '["shop"="optician"]')
#     dist_nearest_mall = nearest_poi_distance(lat, lon, '["shop"="mall"]')
#     dist_nearest_transport = nearest_poi_distance(lat, lon, '["public_transport"]')
#     num_retail_shops = count_pois(lat, lon, '["shop"]')
#     num_restaurants = count_pois(lat, lon, '["amenity"="restaurant"]')
    
#     # Calculate traffic estimates
#     estimated_foot_traffic = estimate_foot_traffic(num_retail_shops, num_restaurants, 1 if dist_nearest_transport != ML_DEFAULT_NUMERIC and dist_nearest_transport < 1 else 0)
#     vehicular_traffic = estimate_vehicular_traffic(lat, lon)
    
#     # Create row data
#     row = [
#         site_id, lat, lon, address_full, city, state, zip_code, locality_type, landmark,
#         int(num_optical_stores) if isinstance(num_optical_stores, (int, str)) and str(num_optical_stores).isdigit() else ML_DEFAULT_NUMERIC,
#         int(num_eye_clinics) if isinstance(num_eye_clinics, (int, str)) and str(num_eye_clinics).isdigit() else ML_DEFAULT_NUMERIC,
#         dist_nearest_optical, dist_nearest_mall, dist_nearest_transport,
#         int(num_retail_shops) if isinstance(num_retail_shops, (int, str)) and str(num_retail_shops).isdigit() else ML_DEFAULT_NUMERIC,
#         int(num_restaurants) if isinstance(num_restaurants, (int, str)) and str(num_restaurants).isdigit() else ML_DEFAULT_NUMERIC,
#         estimated_foot_traffic, vehicular_traffic,
#         property_type, site_area_sqft, asking_rent_per_sqft_inr, floor_level
#     ]
    
#     # Prepare data for ML prediction
#     numerical_cols = ['latitude', 'longitude', 'num_optical_stores_within_1km', 'num_eye_clinics_within_1km', 
#                       'distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km', 
#                       'distance_to_public_transport_km', 'num_retail_shops_within_1km', 
#                       'num_restaurants_within_1km', 'estimated_daily_foot_traffic', 
#                       'vehicular_traffic_count']
#     categorical_cols = ['city', 'state', 'locality_type']
    
#     data_dict = {
#         'latitude': lat, 'longitude': lon, 'city': city, 'state': state, 'locality_type': locality_type,
#         'num_optical_stores_within_1km': row[9], 'num_eye_clinics_within_1km': row[10],
#         'distance_to_nearest_optical_store_km': row[11], 'distance_to_nearest_mall_km': row[12],
#         'distance_to_public_transport_km': row[13], 'num_retail_shops_within_1km': row[14],
#         'num_restaurants_within_1km': row[15], 'estimated_daily_foot_traffic': row[16],
#         'vehicular_traffic_count': row[17]
#     }
#     df = pd.DataFrame([data_dict])
#     df[numerical_cols] = df[numerical_cols].replace(ML_DEFAULT_NUMERIC, np.nan)
    
#     # Validate model features
#     required_features = numerical_cols + categorical_cols
#     missing_features = [col for col in required_features if col not in df.columns]
#     if missing_features:
#         logger.error(f"Missing required features for prediction: {missing_features}")
#         row.append(0.0)
#     else:
#         try:
#             feasibility_score = model.predict(df)[0]
#             row.append(round(feasibility_score, 3))
#         except Exception as e:
#             logger.error(f"Prediction failed: {e}")
#             row.append(0.0)
    
#     # Write to CSV
#     write_to_csv([row], output_csv)
    
#     return row

# # ------------- CSV Writer Function -------------
# def write_to_csv(rows, filename):
#     header = ["site_id", "latitude", "longitude", "address", "city", "state", "zip_code", 
#               "locality_type", "landmark", "num_optical_stores_within_1km", 
#               "num_eye_clinics_within_1km", "distance_to_nearest_optical_store_km", 
#               "distance_to_nearest_mall_km", "distance_to_public_transport_km", 
#               "num_retail_shops_within_1km", "num_restaurants_within_1km", 
#               "estimated_daily_foot_traffic", "vehicular_traffic_count", 
#               "property_type", "site_area_sqft", "asking_rent_per_sqft_inr", 
#               "floor_level", "feasibility_score"]
#     try:
#         file_exists = os.path.exists(filename)
#         mode = 'a' if file_exists else 'w'
#         with open(filename, mode, newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             if not file_exists:
#                 writer.writerow(header)
#             for row in rows:
#                 writer.writerow(row)
#         logger.info(f"Data written to {filename}")
#     except PermissionError:
#         logger.error(f"Permission denied when writing to file: {filename}")
#         raise
#     except Exception as e:
#         logger.error(f"Failed to write to CSV: {e}")
#         raise

# # ------------- Main Execution -------------
# if __name__ == "__main__":
#     logger.info("=== Lenskart Site Feasibility Analysis with ML ===")
    
#     # Data file path
#     data_file = 'expanded_lenskart_site_feasibility_500.csv'
    
#     # Check if we should preprocess data for visualization
#     preprocess_for_viz = input("Do you want to create preprocessed data for visualization? (y/n): ").lower().strip()
#     if preprocess_for_viz in ['y', 'yes']:
#         try:
#             preprocessed_data = preprocess_data_for_visualization(
#                 data_file, 
#                 'preprocessed_data_for_visualization.csv'
#             )
#             logger.info("Preprocessed data created successfully for visualization!")
#         except Exception as e:
#             logger.error(f"Failed to create preprocessed data: {e}")
    
#     # Check if pre-trained model exists
#     model_file = os.getenv('MODEL_FILE_PATH', 'feasibility_model.joblib')
#     if os.path.exists(model_file):
#         logger.info(f"Loading pre-trained model from {model_file}")
#         try:
#             model = joblib.load(model_file)
#         except Exception as e:
#             logger.error(f"Failed to load model: {e}. Training new model.")
#             model = train_and_save_model(
#                 data_file, 
#                 model_file,
#                 save_preprocessed=True,
#                 preprocessed_file='preprocessed_data_for_visualization.csv'
#             )
#     else:
#         logger.info("No pre-trained model found. Training new model.")
#         model = train_and_save_model(
#             data_file, 
#             model_file,
#             save_preprocessed=True,
#             preprocessed_file='preprocessed_data_for_visualization.csv'
#         )
    
#     # User input for prediction
#     address = input("Please enter the address (or press Enter to use lat/lon): ").strip()
#     lat = input("Enter latitude (or press Enter to use address): ").strip()
#     lon = input("Enter longitude (or press Enter to use address): ").strip()
    
#     # Validate input
#     if not address and (not lat or not lon):
#         logger.error("No valid address or lat/lon provided")
#         exit(1)
    
#     # Parse lat/lon if provided
#     lat = float(lat) if lat and lat.replace('.', '').replace('-', '').isdigit() else None
#     lon = float(lon) if lon and lon.replace('.', '').replace('-', '').isdigit() else None
    
#     # Process the address or lat/lon
#     result = lenskart_site_pipeline_with_feasibility(
#         address=address if address else None,
#         lat=lat,
#         lon=lon,
#         property_type="commercial",
#         site_area_sqft=1200,
#         asking_rent_per_sqft_inr=45,
#         floor_level="Ground",
#         model=model,
#         output_csv=OUTPUT_CSV
#     )
    
#     # Display result
#     if result[1] is not None:
#         logger.info(f"\nAddress: {result[3]}")
#         logger.info(f"City: {result[4]}, State: {result[5]}")
#         logger.info(f"Locality Type: {result[7]}, Landmark: {result[8]}")
#         logger.info(f"Feasibility Score: {result[-1]}")
#         logger.info(f"Data saved to '{OUTPUT_CSV}'")
#     else:
#         logger.error(f"Failed to process address: {address}")

import requests
import uuid
import time
import pandas as pd
import numpy as np
import logging
from geopy.distance import geodesic
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ratelimit import limits, sleep_and_retry

# ------------- Config -------------
OVERPASS_URL = "http://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
USER_AGENT = {"User-Agent": "lenskart-site-feasibility-pipeline/2.0"}
ML_DEFAULT_NUMERIC = -999  # Default value for missing numeric fields
ONE_SECOND = 1  # Rate limit period for API calls
CSV_FILE = 'expanded_lenskart_site_feasibility_500.csv'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------- Rate-limited API Call -------------
@sleep_and_retry
@limits(calls=1, period=ONE_SECOND)
def make_api_request(url, params=None, data=None, headers=None):
    try:
        response = requests.get(url, params=params, headers=headers) if params else requests.post(url, data=data, headers=headers)
        response.raise_for_status()
        logger.info(f"API request successful: {url}")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {url}, Error: {e}, Response: {response.text if 'response' in locals() else 'N/A'}")
        return None

# ------------- Train and Save Model -------------
def train_and_save_model(data_file, model_file='feasibility_model.joblib', save_preprocessed=True, preprocessed_file='preprocessed_data_for_visualization.csv'):
    logger.info("Training the model...")
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        logger.error(f"Input file not found: {data_file}")
        raise
    except PermissionError:
        logger.error(f"Permission denied when reading file: {data_file}")
        raise
    
    original_df = df.copy()
    columns_to_drop = ['site_id', 'address', 'landmark', 'zip_code', 'property_type', 
                       'site_area_sqft', 'asking_rent_per_sqft_inr', 'floor_level']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
    
    numerical_cols = ['latitude', 'longitude', 'num_optical_stores_within_1km', 'num_eye_clinics_within_1km',
                      'distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km',
                      'distance_to_public_transport_km', 'num_retail_shops_within_1km',
                      'num_restaurants_within_1km', 'estimated_daily_foot_traffic',
                      'vehicular_traffic_count']
    missing_cols = [col for col in numerical_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing expected columns: {missing_cols}")
        raise ValueError(f"Missing expected columns: {missing_cols}")
    
    df[numerical_cols] = df[numerical_cols].replace(-999, np.nan)
    categorical_cols = ['city', 'state', 'locality_type']
    
    if save_preprocessed:
        try:
            viz_df = df.copy()
            viz_df['site_id'] = original_df['site_id']
            viz_df['address'] = original_df['address']
            viz_df['landmark'] = original_df['landmark']
            viz_df['zip_code'] = original_df['zip_code']
            viz_df['property_type'] = original_df['property_type']
            viz_df['site_area_sqft'] = original_df['site_area_sqft']
            viz_df['asking_rent_per_sqft_inr'] = original_df['asking_rent_per_sqft_inr']
            viz_df['floor_level'] = original_df['floor_level']
            id_cols = ['site_id', 'address', 'landmark', 'zip_code', 'property_type', 
                      'site_area_sqft', 'asking_rent_per_sqft_inr', 'floor_level']
            other_cols = [col for col in viz_df.columns if col not in id_cols]
            viz_df = viz_df[id_cols + other_cols]
            viz_df.to_csv(preprocessed_file, index=False)
            logger.info(f"Preprocessed data saved to {preprocessed_file} for visualization purposes")
        except Exception as e:
            logger.warning(f"Failed to save preprocessed data: {e}")
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    X = df.drop('feasibility_score', axis=1)
    y = df['feasibility_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f'Model trained - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}')
    
    try:
        joblib.dump(pipeline, model_file)
        logger.info(f"Model saved to {model_file}")
    except PermissionError:
        logger.error(f"Permission denied when saving model to {model_file}")
        raise
    return pipeline

# ------------- Preprocess Data for Visualization -------------
def preprocess_data_for_visualization(input_file, output_file='preprocessed_data_for_visualization.csv'):
    logger.info(f"Preprocessing {input_file} for visualization...")
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from {input_file}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except PermissionError:
        logger.error(f"Permission denied when reading file: {input_file}")
        raise
        
    processed_df = df.copy()
    numerical_cols = ['latitude', 'longitude', 'num_optical_stores_within_1km', 'num_eye_clinics_within_1km',
                      'distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km',
                      'distance_to_public_transport_km', 'num_retail_shops_within_1km',
                      'num_restaurants_within_1km', 'estimated_daily_foot_traffic',
                      'vehicular_traffic_count']
    missing_cols = [col for col in numerical_cols if col not in processed_df.columns]
    if missing_cols:
        logger.error(f"Missing expected columns: {missing_cols}")
        raise ValueError(f"Missing expected columns: {missing_cols}")
    
    for col in numerical_cols:
        processed_df[col] = processed_df[col].replace(-999, np.nan)
        logger.info(f"Processed column {col}: {processed_df[col].isnull().sum()} missing values")
    
    processed_df['has_missing_coordinates'] = processed_df['latitude'].isnull() | processed_df['longitude'].isnull()
    processed_df['has_missing_poi_data'] = processed_df[['num_optical_stores_within_1km', 'num_eye_clinics_within_1km']].isnull().any(axis=1)
    processed_df['has_missing_distance_data'] = processed_df[['distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km', 'distance_to_public_transport_km']].isnull().any(axis=1)
    processed_df['has_missing_traffic_data'] = processed_df[['estimated_daily_foot_traffic', 'vehicular_traffic_count']].isnull().any(axis=1)
    processed_df['data_quality_score'] = processed_df[numerical_cols].notna().sum(axis=1) / len(numerical_cols) if len(numerical_cols) > 0 else 0
    
    if 'feasibility_score' in processed_df.columns:
        processed_df['feasibility_category'] = pd.cut(
            processed_df['feasibility_score'], 
            bins=[0, 3, 5, 7, 10], 
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
    
    if 'state' in processed_df.columns:
        north_states = ['Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Himachal Pradesh', 'Uttarakhand']
        south_states = ['Karnataka', 'Tamil Nadu', 'Andhra Pradesh', 'Telangana', 'Kerala']
        west_states = ['Maharashtra', 'Gujarat', 'Goa']
        east_states = ['West Bengal', 'Odisha', 'Jharkhand', 'Bihar']
        central_states = ['Madhya Pradesh', 'Chhattisgarh', 'Rajasthan']
        
        def get_region(state):
            if state in north_states: return 'North'
            elif state in south_states: return 'South'
            elif state in west_states: return 'West'
            elif state in east_states: return 'East'
            elif state in central_states: return 'Central'
            else: return 'Other'
        
        processed_df['region'] = processed_df['state'].apply(get_region)
    
    try:
        processed_df.to_csv(output_file, index=False)
        logger.info(f"Preprocessed data saved to {output_file}")
        logger.info(f"Preprocessed data shape: {processed_df.shape}")
    except PermissionError:
        logger.error(f"Permission denied when saving file: {output_file}")
        raise
    
    print("\n=== Preprocessing Summary ===")
    print(f"Total records: {len(processed_df)}")
    print(f"Records with missing coordinates: {processed_df['has_missing_coordinates'].sum()}")
    print(f"Records with missing POI data: {processed_df['has_missing_poi_data'].sum()}")
    print(f"Records with missing distance data: {processed_df['has_missing_distance_data'].sum()}")
    print(f"Records with missing traffic data: {processed_df['has_missing_traffic_data'].sum()}")
    print(f"Average data quality score: {processed_df['data_quality_score'].mean():.2f}")
    if 'feasibility_category' in processed_df.columns:
        print("\nFeasibility Score Distribution:")
        print(processed_df['feasibility_category'].value_counts())
    if 'region' in processed_df.columns:
        print("\nRegional Distribution:")
        print(processed_df['region'].value_counts())
    
    return processed_df

# ------------- Geocoding Functions -------------
def sleep():
    time.sleep(1.2)

def geocode_address(address):
    logger.info(f"Geocoding address: {address}")
    address_parts = [part.strip() for part in address.split(',')]
    cleaned_parts = [part for part in address_parts if not any(kw in part.lower() for kw in ['shop no', 'unit no', 'near'])]
    cleaned_address = ', '.join(cleaned_parts)
    
    # Define known coordinates for major cities as fallback
    city_fallbacks = {
        'bengaluru': {'lat': 12.9716, 'lon': 77.5946},
        'bangalore': {'lat': 12.9716, 'lon': 77.5946},  # Handle alternate spelling
        'mumbai': {'lat': 19.0760, 'lon': 72.8777},
        'delhi': {'lat': 28.7041, 'lon': 77.1025}
    }
    
    street, city, state, postcode = '', '', '', ''
    for part in cleaned_parts:
        part_lower = part.lower()
        if any(kw in part_lower for kw in ['road', 'street', 'nagar', 'complex', 'plot']):
            street = part if not street else f"{street}, {part}"
        elif any(kw in part_lower for kw in ['indore', 'mumbai', 'delhi', 'bangalore', 'bengaluru', 'kolkata', 'chennai']):
            city = part_lower
        elif any(kw in part_lower for kw in ['madhya pradesh', 'maharashtra', 'karnataka', 'west bengal', 'tamil nadu']):
            state = part
        elif part.replace(' ', '').isdigit():
            postcode = part
    
    # Try structured query
    params = {k: v for k, v in {'street': street, 'city': city, 'state': state, 'postalcode': postcode, 'country': 'India', 'format': 'json', 'limit': 1}.items() if v}
    try:
        r = make_api_request(f"{NOMINATIM_URL}/search", params=params, headers=USER_AGENT)
        if r:
            r = r.json()
            if r and isinstance(r, list) and len(r) > 0 and 'lat' in r[0] and 'lon' in r[0]:
                logger.info(f"Geocoded {address} to ({r[0]['lat']}, {r[0]['lon']})")
                return float(r[0]['lat']), float(r[0]['lon'])
    except Exception as e:
        logger.warning(f"Structured query failed: {e}")
    
    # Try free-form query
    params = {'q': cleaned_address, 'format': 'json', 'limit': 1}
    try:
        r = make_api_request(f"{NOMINATIM_URL}/search", params=params, headers=USER_AGENT)
        if r:
            r = r.json()
            if r and isinstance(r, list) and len(r) > 0 and 'lat' in r[0] and 'lon' in r[0]:
                logger.info(f"Geocoded {address} to ({r[0]['lat']}, {r[0]['lon']})")
                return float(r[0]['lat']), float(r[0]['lon'])
    except Exception as e:
        logger.warning(f"Free-form query failed: {e}")
    
    # Fallback to known city coordinates
    city_lower = city.lower() if city else ''
    if city_lower in city_fallbacks:
        logger.warning(f"Using fallback coordinates for {city_lower}: {city_fallbacks[city_lower]}")
        return city_fallbacks[city_lower]['lat'], city_fallbacks[city_lower]['lon']
    
    # Try minimal address (city, state)
    if city and state:
        minimal_address = f"{city}, {state}"
        logger.warning(f"Falling back to minimal address: {minimal_address}")
        params = {'q': minimal_address, 'format': 'json', 'limit': 1}
        try:
            r = make_api_request(f"{NOMINATIM_URL}/search", params=params, headers=USER_AGENT)
            if r:
                r = r.json()
                if r and isinstance(r, list) and len(r) > 0 and 'lat' in r[0] and 'lon' in r[0]:
                    logger.info(f"Geocoded {minimal_address} to ({r[0]['lat']}, {r[0]['lon']})")
                    return float(r[0]['lat']), float(r[0]['lon'])
        except Exception as e:
            logger.warning(f"Minimal address query failed: {e}")
    
    logger.error(f"Geocoding failed for {address}")
    return None, None

def reverse_geocode(lat, lon):
    params = {'lat': lat, 'lon': lon, 'format': 'json'}
    try:
        r = make_api_request(f"{NOMINATIM_URL}/reverse", params=params, headers=USER_AGENT)
        if r:
            r = r.json()
            addr = r.get('address', {})
            return (r.get('display_name', ''), addr.get('city', '') or addr.get('town', '') or addr.get('village', ''), addr.get('state', ''), addr.get('postcode', ''))
    except Exception as e:
        logger.error(f"Reverse geocoding failed: {e}")
        return '', '', '', ''

def get_locality_type(lat, lon):
    params = {'lat': lat, 'lon': lon, 'format': 'json'}
    try:
        r = make_api_request(f"{NOMINATIM_URL}/reverse", params=params, headers=USER_AGENT)
        if r:
            r = r.json()
            t = r.get('address', {}).get('city', '') or r.get('address', {}).get('town', '') or r.get('address', {}).get('village', '')
            return "Urban" if t else "Rural" if r.get('address', {}).get('hamlet', '') else "Semi-Urban"
    except Exception as e:
        logger.error(f"Locality type fetch failed: {e}")
        return "Semi-Urban"

def get_better_landmark(lat, lon):
    priorities = [('railway_station', 'amenity'), ('subway_entrance', 'railway'), ('mall', 'shop'), ('hospital', 'amenity'), ('school', 'amenity'), ('bank', 'amenity'), ('park', 'leisure'), ('hotel', 'tourism'), ('marketplace', 'amenity'), ('restaurant', 'amenity'), ('parking', 'amenity'), ('bus_station', 'amenity'), ('cinema', 'amenity'), ('college', 'amenity'), ('university', 'amenity')]
    for key, tag_type in priorities:
        query = f"[out:json][timeout:25];(node(around:400,{lat},{lon})[{tag_type}=\"{key}\"][name];way(around:400,{lat},{lon})[{tag_type}=\"{key}\"][name];rel(around:400,{lat},{lon})[{tag_type}=\"{key}\"][name];);out center 1;"
        try:
            r = make_api_request(OVERPASS_URL, data={'data': query}, headers=USER_AGENT)
            if r:
                r = r.json()
                if r.get('elements', []):
                    name = r['elements'][0].get('tags', {}).get('name')
                    if name:
                        return name
        except Exception as e:
            logger.warning(f"Landmark query failed for {key}: {e}")
    return "No landmark found"

def count_pois(lat, lon, osm_query):
    query = f"[out:json][timeout:25];(node{osm_query}(around:2000,{lat},{lon});way{osm_query}(around:2000,{lat},{lon});rel{osm_query}(around:2000,{lat},{lon}););out count;"
    try:
        r = make_api_request(OVERPASS_URL, data={'data': query}, headers=USER_AGENT)
        if r:
            r = r.json()
            count = r.get('elements', [{}])[0].get('tags', {}).get('total', 0)
            logger.info(f"POI count for {osm_query} at ({lat}, {lon}): {count}")
            return count
        else:
            logger.warning(f"No response from Overpass API for {osm_query}")
            return 0  # Use 0 instead of -999
    except Exception as e:
        logger.error(f"POI count failed for {osm_query}: {e}")
        return 0

def nearest_poi_distance(lat, lon, osm_query):
    query = f"[out:json][timeout:25];(node{osm_query}(around:10000,{lat},{lon});way{osm_query}(around:10000,{lat},{lon});rel{osm_query}(around:10000,{lat},{lon}););out center;"
    try:
        r = make_api_request(OVERPASS_URL, data={'data': query}, headers=USER_AGENT)
        if r:
            r = r.json()
            min_dist = float('inf')
            nearest_poi = None
            for el in r.get('elements', []):
                poi_lat = el.get('lat') or el.get('center', {}).get('lat')
                poi_lon = el.get('lon') or el.get('center', {}).get('lon')
                if poi_lat and poi_lon:
                    d = geodesic((lat, lon), (poi_lat, poi_lon)).km
                    if d < min_dist:
                        min_dist = d
                        nearest_poi = {
                            'lat': poi_lat,
                            'lon': poi_lon,
                            'name': el.get('tags', {}).get('name', 'Unknown')
                        }
            if nearest_poi:
                logger.info(f"Nearest POI for {osm_query} at ({lat}, {lon}): {min_dist} km, {nearest_poi}")
                return round(min_dist, 3), nearest_poi
            else:
                logger.warning(f"No POIs found for {osm_query} within 10km")
                return 5.0, None  # Use 5.0 km as a reasonable default
        else:
            logger.warning(f"No response from Overpass API for {osm_query}")
            return 5.0, None
    except Exception as e:
        logger.error(f"Nearest POI distance failed for {osm_query}: {e}")
        return 5.0, None

def estimate_foot_traffic(num_retail, num_restaurants, num_transport):
    try:
        num_retail = int(float(num_retail)) if num_retail is not None and str(num_retail).replace('.', '').replace('-', '').isdigit() else 0
        num_restaurants = int(float(num_restaurants)) if num_restaurants is not None and str(num_restaurants).replace('.', '').replace('-', '').isdigit() else 0
        num_transport = int(float(num_transport)) if num_transport is not None and str(num_transport).replace('.', '').replace('-', '').isdigit() else 0
        foot_traffic = num_retail * 2 + num_restaurants * 2 + num_transport * 5
        logger.info(f"Estimated foot traffic: {foot_traffic} (retail: {num_retail}, restaurants: {num_restaurants}, transport: {num_transport})")
        return foot_traffic
    except (ValueError, TypeError):
        logger.warning("Invalid input for foot traffic estimation, returning default value")
        return ML_DEFAULT_NUMERIC

def estimate_vehicular_traffic(lat, lon):
    query = f"[out:json][timeout:25];(way(around:200,{lat},{lon})[highway~\"primary|secondary|trunk\"];);out;"
    try:
        r = make_api_request(OVERPASS_URL, data={'data': query}, headers=USER_AGENT)
        if r:
            r = r.json()
            count = len(r.get('elements', [])) * 50
            logger.info(f"Estimated vehicular traffic at ({lat}, {lon}): {count}")
            return count
        else:
            logger.warning(f"No response from Overpass API for vehicular traffic")
            return ML_DEFAULT_NUMERIC
    except Exception as e:
        logger.error(f"Vehicular traffic query failed: {e}")
        return ML_DEFAULT_NUMERIC

def write_to_csv(data, csv_file=CSV_FILE):
    try:
        # Flatten nearest_pois for CSV
        data_flat = data.copy()
        data_flat.pop('nearest_pois', None)  # Remove nested list
        df = pd.DataFrame([data_flat])
        
        # Ensure all expected columns are present
        expected_cols = ['site_id', 'latitude', 'longitude', 'address', 'city', 'state', 'zip_code', 
                         'locality_type', 'landmark', 'num_optical_stores_within_1km', 
                         'num_eye_clinics_within_1km', 'distance_to_nearest_optical_store_km', 
                         'distance_to_nearest_mall_km', 'distance_to_public_transport_km', 
                         'num_retail_shops_within_1km', 'num_restaurants_within_1km', 
                         'estimated_daily_foot_traffic', 'vehicular_traffic_count', 
                         'property_type', 'site_area_sqft', 'asking_rent_per_sqft_inr', 
                         'floor_level', 'feasibility_score']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Append to CSV
        df = df[expected_cols]  # Reorder columns
        df.to_csv(csv_file, mode='a', index=False, header=not pd.io.common.file_exists(csv_file))
        logger.info(f"Data appended to {csv_file}")
    except Exception as e:
        logger.error(f"Failed to write to CSV {csv_file}: {e}")

# ------------- Main Pipeline Function with ML Prediction -------------
def lenskart_site_pipeline_with_feasibility(address=None, lat=None, lon=None, property_type=None, site_area_sqft=None, asking_rent_per_sqft_inr=None, floor_level=None, model=None):
    site_id = str(uuid.uuid4())
    
    property_type = property_type.strip() if property_type and isinstance(property_type, str) and property_type.strip() else "commercial"
    site_area_sqft = int(float(site_area_sqft)) if site_area_sqft and str(site_area_sqft).replace('.', '').replace('-', '').isdigit() and float(site_area_sqft) > 0 else ML_DEFAULT_NUMERIC
    asking_rent_per_sqft_inr = int(float(asking_rent_per_sqft_inr)) if asking_rent_per_sqft_inr and str(asking_rent_per_sqft_inr).replace('.', '').replace('-', '').isdigit() and float(asking_rent_per_sqft_inr) > 0 else ML_DEFAULT_NUMERIC
    floor_level = floor_level.strip() if floor_level and isinstance(floor_level, str) and floor_level.strip() else "Ground"
    
    if address and (lat is None or lon is None):
        lat, lon = geocode_address(address)
        if lat is None or lon is None:
            logger.error(f"Geocoding failed for {address}")
            return {'error': 'Geocoding failed'}
    elif lat is None or lon is None:
        logger.error("No address or lat/lon provided")
        return {'error': 'No location provided'}

    address_full, city, state, zip_code = reverse_geocode(lat, lon)
    locality_type = get_locality_type(lat, lon)
    landmark = get_better_landmark(lat, lon)
    
    # Updated OSM queries
    num_optical_stores = count_pois(lat, lon, '["shop"~"optician|optics|eyewear"]')
    num_eye_clinics = count_pois(lat, lon, '["healthcare"~"clinic|hospital"]["healthcare:speciality"~"optometry|ophthalmology"]')
    dist_nearest_optical, nearest_optical = nearest_poi_distance(lat, lon, '["shop"~"optician|optics|eyewear"]')
    dist_nearest_mall, nearest_mall = nearest_poi_distance(lat, lon, '["shop"~"mall|shopping_centre|department_store"]')
    dist_nearest_transport, nearest_transport = nearest_poi_distance(lat, lon, '["public_transport"]')
    num_retail_shops = count_pois(lat, lon, '["shop"]')
    num_restaurants = count_pois(lat, lon, '["amenity"~"restaurant|cafe"]')
    
    estimated_foot_traffic = estimate_foot_traffic(num_retail_shops, num_restaurants, 1 if dist_nearest_transport != ML_DEFAULT_NUMERIC and dist_nearest_transport < 1 else 0)
    vehicular_traffic = estimate_vehicular_traffic(lat, lon)
    
    numerical_cols = ['latitude', 'longitude', 'num_optical_stores_within_1km', 'num_eye_clinics_within_1km', 
                      'distance_to_nearest_optical_store_km', 'distance_to_nearest_mall_km', 
                      'distance_to_public_transport_km', 'num_retail_shops_within_1km', 
                      'num_restaurants_within_1km', 'estimated_daily_foot_traffic', 
                      'vehicular_traffic_count']
    categorical_cols = ['city', 'state', 'locality_type']
    
    data_dict = {
        'latitude': lat, 'longitude': lon, 'city': city, 'state': state, 'locality_type': locality_type,
        'num_optical_stores_within_1km': num_optical_stores if num_optical_stores != ML_DEFAULT_NUMERIC else np.nan,
        'num_eye_clinics_within_1km': num_eye_clinics if num_eye_clinics != ML_DEFAULT_NUMERIC else np.nan,
        'distance_to_nearest_optical_store_km': dist_nearest_optical if dist_nearest_optical != ML_DEFAULT_NUMERIC else np.nan,
        'distance_to_nearest_mall_km': dist_nearest_mall if dist_nearest_mall != ML_DEFAULT_NUMERIC else np.nan,
        'distance_to_public_transport_km': dist_nearest_transport if dist_nearest_transport != ML_DEFAULT_NUMERIC else np.nan,
        'num_retail_shops_within_1km': num_retail_shops if num_retail_shops != ML_DEFAULT_NUMERIC else np.nan,
        'num_restaurants_within_1km': num_restaurants if num_restaurants != ML_DEFAULT_NUMERIC else np.nan,
        'estimated_daily_foot_traffic': estimated_foot_traffic if estimated_foot_traffic != ML_DEFAULT_NUMERIC else np.nan,
        'vehicular_traffic_count': vehicular_traffic if vehicular_traffic != ML_DEFAULT_NUMERIC else np.nan
    }
    df = pd.DataFrame([data_dict])
    
    required_features = numerical_cols + categorical_cols
    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        logger.error(f"Missing required features for prediction: {missing_features}")
        feasibility_score = 0.0
    else:
        try:
            feasibility_score = model.predict(df)[0]
            feasibility_score = round(feasibility_score, 3)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            feasibility_score = 0.0
    
    nearest_pois = []
    if nearest_optical:
        nearest_pois.append({'type': 'optical_store', **nearest_optical})
    if nearest_mall:
        nearest_pois.append({'type': 'mall', **nearest_mall})
    if nearest_transport:
        nearest_pois.append({'type': 'public_transport', **nearest_transport})
    
    # Prepare data for CSV
    data_for_csv = {
        'site_id': site_id,
        'latitude': lat,
        'longitude': lon,
        'address': address_full,
        'city': city,
        'state': state,
        'zip_code': zip_code,
        'locality_type': locality_type,
        'landmark': landmark,
        'num_optical_stores_within_1km': num_optical_stores,
        'num_eye_clinics_within_1km': num_eye_clinics,
        'distance_to_nearest_optical_store_km': dist_nearest_optical,
        'distance_to_nearest_mall_km': dist_nearest_mall,
        'distance_to_public_transport_km': dist_nearest_transport,
        'num_retail_shops_within_1km': num_retail_shops,
        'num_restaurants_within_1km': num_restaurants,
        'estimated_daily_foot_traffic': estimated_foot_traffic,
        'vehicular_traffic_count': vehicular_traffic,
        'property_type': property_type,
        'site_area_sqft': site_area_sqft,
        'asking_rent_per_sqft_inr': asking_rent_per_sqft_inr,
        'floor_level': floor_level,
        'feasibility_score': feasibility_score
    }
    
    # Write to CSV
    write_to_csv(data_for_csv, CSV_FILE)
    
    # Return only the feasibility score
    return {'feasibility_score': feasibility_score}
    
# ------------- Main Execution -------------
if __name__ == "__main__":
    logger.info("=== Lenskart Site Feasibility Analysis with ML ===")
    data_file = 'expanded_lenskart_site_feasibility_500.csv'
    
    preprocess_for_viz = input("Do you want to create preprocessed data for visualization? (y/n): ").lower().strip()
    if preprocess_for_viz in ['y', 'yes']:
        try:
            preprocessed_data = preprocess_data_for_visualization(
                data_file, 
                'preprocessed_data_for_visualization.csv'
            )
            logger.info("Preprocessed data created successfully for visualization!")
        except Exception as e:
            logger.error(f"Failed to create preprocessed data: {e}")
    
    model_file = 'feasibility_model.joblib'
    if pd.io.common.file_exists(model_file):
        logger.info(f"Loading pre-trained model from {model_file}")
        try:
            model = joblib.load(model_file)
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Training new model.")
            model = train_and_save_model(
                data_file, 
                model_file,
                save_preprocessed=True,
                preprocessed_file='preprocessed_data_for_visualization.csv'
            )
    else:
        logger.info("No pre-trained model found. Training new model.")
        model = train_and_save_model(
            data_file, 
            model_file,
            save_preprocessed=True,
            preprocessed_file='preprocessed_data_for_visualization.csv'
        )
    
    address = input("Please enter the address (or press Enter to use lat/lon): ").strip()
    lat = input("Enter latitude (or press Enter to use address): ").strip()
    lon = input("Enter longitude (or press Enter to use address): ").strip()
    
    if not address and (not lat or not lon):
        logger.error("No valid address or lat/lon provided")
        exit(1)
    
    lat = float(lat) if lat and lat.replace('.', '').replace('-', '').isdigit() else None
    lon = float(lon) if lon and lon.replace('.', '').replace('-', '').isdigit() else None
    
    result = lenskart_site_pipeline_with_feasibility(
        address=address if address else None,
        lat=lat,
        lon=lon,
        property_type="commercial",
        site_area_sqft=1200,
        asking_rent_per_sqft_inr=45,
        floor_level="Ground",
        model=model
    )
    
    if 'error' not in result:
        logger.info(f"Feasibility Score: {result['feasibility_score']}")
    else:
        logger.error(f"Failed to process: {result['error']}")