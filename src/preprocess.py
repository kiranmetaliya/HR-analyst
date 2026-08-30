import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Robust smoothed target encoder for high-cardinality variables.
    Learns mappings on training data and applies them, preventing target leakage.
    Formula: (count * mean + smoothing * global_mean) / (count + smoothing)
    """
    def __init__(self, cols=None, smoothing=10.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mappings = {}
        self.global_mean = 0.0

    def fit(self, X, y):
        self.global_mean = y.mean()
        X_temp = X.copy()
        X_temp['target'] = y
        
        self.mappings = {}
        cols_to_encode = self.cols if self.cols is not None else X.select_dtypes(include=['object', 'category']).columns
        
        for col in cols_to_encode:
            # Convert column to string to ensure consistent mapping
            col_series = X_temp[col].astype(str)
            stats = X_temp.groupby(col_series)['target'].agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
            self.mappings[col] = smoothed.to_dict()
            
        return self

    def transform(self, X):
        X_out = X.copy()
        cols_to_encode = self.cols if self.cols is not None else X.select_dtypes(include=['object', 'category']).columns
        
        for col in cols_to_encode:
            col_series = X_out[col].astype(str)
            mapping = self.mappings.get(col, {})
            # Map values, fill missing/unseen values with the global mean
            X_out[col] = col_series.map(mapping).fillna(self.global_mean)
            
        return X_out

class BookingPreprocessor:
    def __init__(self):
        self.high_cardinality_cols = ['country', 'agent', 'company']
        self.low_cardinality_cols = ['hotel', 'meal', 'market_segment', 'distribution_channel', 
                                     'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']
        
        self.month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        self.target_encoder = SmoothedTargetEncoder(cols=self.high_cardinality_cols, smoothing=10.0)
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        
        # We will keep track of column names for the output DataFrame
        self.feature_names_ = None

    def fit_transform(self, X, y):
        # 1. Cleaning & Imputation (inline to avoid pipeline state issues)
        X_clean = self._clean_and_impute(X)
        
        # 2. Fit and transform target encoder on high-cardinality features
        X_encoded = X_clean.copy()
        X_encoded[self.high_cardinality_cols] = self.target_encoder.fit_transform(X_clean[self.high_cardinality_cols], y)
        
        # 3. Fit and transform one-hot encoder on low-cardinality features
        ohe_features = self.one_hot_encoder.fit_transform(X_encoded[self.low_cardinality_cols])
        ohe_cols = [f"{col}_{val}" for col, vals in zip(self.low_cardinality_cols, self.one_hot_encoder.categories_) for val in vals]
        
        # Drop the original low-cardinality categorical columns
        X_encoded = X_encoded.drop(columns=self.low_cardinality_cols)
        
        # 4. Construct final feature dataframe
        ohe_df = pd.DataFrame(ohe_features, columns=ohe_cols, index=X_encoded.index)
        X_final = pd.concat([X_encoded, ohe_df], axis=1)
        
        # 5. Fit & scale numerical and target-encoded features
        scaled_features = self.scaler.fit_transform(X_final)
        self.feature_names_ = list(X_final.columns)
        
        return pd.DataFrame(scaled_features, columns=self.feature_names_, index=X.index)

    def transform(self, X):
        # 1. Cleaning & Imputation
        X_clean = self._clean_and_impute(X)
        
        # 2. Target Encoding
        X_encoded = X_clean.copy()
        X_encoded[self.high_cardinality_cols] = self.target_encoder.transform(X_clean[self.high_cardinality_cols])
        
        # 3. One-hot Encoding
        ohe_features = self.one_hot_encoder.transform(X_encoded[self.low_cardinality_cols])
        ohe_cols = [f"{col}_{val}" for col, vals in zip(self.low_cardinality_cols, self.one_hot_encoder.categories_) for val in vals]
        
        X_encoded = X_encoded.drop(columns=self.low_cardinality_cols)
        
        # 4. Construct final dataframe
        ohe_df = pd.DataFrame(ohe_features, columns=ohe_cols, index=X_encoded.index)
        X_final = pd.concat([X_encoded, ohe_df], axis=1)
        
        # 5. Scale features
        scaled_features = self.scaler.transform(X_final)
        
        return pd.DataFrame(scaled_features, columns=self.feature_names_, index=X.index)

    def _clean_and_impute(self, df):
        X = df.copy()
        
        # Secure imputations
        X['children'] = X['children'].fillna(0.0)
        X['country'] = X['country'].fillna('Unknown')
        X['agent'] = X['agent'].fillna(0.0).astype(int).astype(str)
        X['company'] = X['company'].fillna(0.0).astype(int).astype(str)
        
        # Ordinal month encoding
        X['arrival_date_month_num'] = X['arrival_date_month'].map(self.month_map).fillna(1).astype(int)
        X = X.drop(columns=['arrival_date_month'])
        
        # Feature engineering
        X['total_stay'] = X['stays_in_weekend_nights'] + X['stays_in_week_nights']
        X['total_guests'] = X['adults'] + X['children'] + X['babies']
        X['has_special_requests'] = (X['total_of_special_requests'] > 0).astype(int)
        X['is_extreme_lead_time'] = (X['lead_time'] > 200).astype(int)
        
        # Drop target leaks
        leak_cols = ['reservation_status', 'reservation_status_date']
        for col in leak_cols:
            if col in X.columns:
                X = X.drop(columns=[col])
                
        return X

def run_preprocessing():
    print("Starting preprocessing pipeline...")
    raw_data_path = "data/raw/hotel_bookings.csv"
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found at {raw_data_path}. Run download_data.py first.")
        
    df = pd.read_csv(raw_data_path)
    
    # Split features and target
    X = df.drop(columns=['is_canceled'])
    y = df['is_canceled']
    
    # Train-test split (80/20 stratified split)
    print("Performing 80/20 stratified train-test split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Fit preprocessor on train set and transform both
    preprocessor = BookingPreprocessor()
    print("Fitting preprocessor and transforming training set...")
    X_train_preprocessed = preprocessor.fit_transform(X_train_raw, y_train)
    print("Transforming testing set...")
    X_test_preprocessed = preprocessor.transform(X_test_raw)
    
    # Save processed datasets
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    X_train_preprocessed.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test_preprocessed.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    # Save fitted preprocessor
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    
    print(f"Preprocessed datasets saved to {processed_dir}/")
    print(f"Fitted preprocessor object saved to {models_dir}/preprocessor.joblib")
    print(f"Preprocessed features count: {X_train_preprocessed.shape[1]}")

if __name__ == "__main__":
    run_preprocessing()
