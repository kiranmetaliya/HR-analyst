import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shap

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import preprocessing classes to resolve deserialization
try:
    from src.preprocess import BookingPreprocessor, SmoothedTargetEncoder
except ImportError:
    from preprocess import BookingPreprocessor, SmoothedTargetEncoder

# Bind classes to __main__ to allow joblib to deserialize them
import __main__
__main__.BookingPreprocessor = BookingPreprocessor
__main__.SmoothedTargetEncoder = SmoothedTargetEncoder

def load_resources():
    models_dir = "models"
    processed_dir = "data/processed"
    
    lgb_model = joblib.load(os.path.join(models_dir, "lightgbm.joblib"))
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    
    return lgb_model, preprocessor, X_test, y_test

def get_shap_values(model, X_sample):
    """Computes SHAP values, handle differences in SHAP versions/outputs."""
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)
    
    # Extract baseline expectation value
    if hasattr(explainer, 'expected_value'):
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            base_value = base_value[1]
    else:
        base_value = 0.5
        
    # Standardize SHAP values for class 1 (cancellation)
    if isinstance(shap_values_raw, list):
        # TreeExplainer returns [shap_values_class0, shap_values_class1] for binary classification
        shap_values = shap_values_raw[1]
    elif isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
        # Array shape: (samples, features, classes)
        shap_values = shap_values_raw[:, :, 1]
    else:
        # Single array or Explanation object
        shap_values = shap_values_raw
        
    # If shap_values is a SHAP Explanation object (SHAP >= 0.40), extract values
    if hasattr(shap_values, 'values'):
        shap_values_arr = shap_values.values
        # If explanation is 3D, extract positive class
        if len(shap_values_arr.shape) == 3:
            shap_values_arr = shap_values_arr[:, :, 1]
            base_value = shap_values.base_values[1] if isinstance(shap_values.base_values, (list, np.ndarray)) else shap_values.base_values
        else:
            base_value = shap_values.base_values
        shap_values = shap_values_arr
        
    return explainer, shap_values, base_value

def generate_global_explanations(model, X_test, sample_size=500):
    print(f"Sampling {sample_size} test points for global SHAP analysis...")
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    
    explainer, shap_values, base_value = get_shap_values(model, X_sample)
    
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    
    # Create the SHAP summary plot
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("LightGBM Feature Importance (SHAP Summary Plot) - Cancellation Prediction", fontsize=14, pad=20)
    plt.tight_layout()
    
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    plot_path = os.path.join(outputs_dir, "shap_summary.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Global SHAP summary plot saved to: {plot_path}")

def explain_local_prediction(model, X_test, y_test, index=None):
    if index is None:
        # Default: Pick one high-risk booking (which canceled) and one low-risk booking (which didn't)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Canceled sample
        canceled_indices = np.where((y_test == 1) & (y_probs > 0.8))[0]
        canceled_idx = canceled_indices[0] if len(canceled_indices) > 0 else 0
        
        # Not-canceled sample
        kept_indices = np.where((y_test == 0) & (y_probs < 0.2))[0]
        kept_idx = kept_indices[0] if len(kept_indices) > 0 else 1
        
        indices_to_explain = [("High Risk (Actual: Canceled)", X_test.index[canceled_idx]),
                              ("Low Risk (Actual: Checked-out)", X_test.index[kept_idx])]
    else:
        if index not in X_test.index:
            raise KeyError(f"Index {index} is not present in the test features dataframe.")
        indices_to_explain = [(f"User Specified Index {index}", index)]
        
    for label, idx in indices_to_explain:
        # Retrieve sample row
        row = X_test.loc[[idx]]
        prob = model.predict_proba(row)[0, 1]
        actual = y_test.loc[idx]
        
        # Compute SHAP values for this specific row
        explainer, shap_val, base_val = get_shap_values(model, row)
        
        # Handle SHAP values list vs array
        if len(shap_val.shape) == 2:
            shap_val = shap_val[0]
            
        # Create a dataframe of features and their SHAP values
        contributions = pd.DataFrame({
            'Feature': X_test.columns,
            'FeatureValue': row.values[0],
            'SHAPValue': shap_val
        })
        
        # Sort by absolute SHAP value
        contributions['AbsSHAP'] = contributions['SHAPValue'].abs()
        contributions = contributions.sort_values(by='AbsSHAP', ascending=False).drop(columns=['AbsSHAP'])
        
        print("\n" + "=" * 60)
        print(f"Local Explanation: {label}")
        print(f"Test Row Index: {idx}")
        print(f"Model Predicted Probability of Cancellation: {prob:.4f} (Raw margin base value: {base_val:.4f})")
        print(f"Actual Outcome: {'Canceled' if actual == 1 else 'Checked-out'}")
        print("-" * 60)
        print(f"{'Feature Name':<35} | {'Scaled Value':<12} | {'SHAP Contribution':<18}")
        print("-" * 60)
        for _, r in contributions.head(8).iterrows():
            print(f"{r['Feature']:<35} | {r['FeatureValue']:<12.4f} | {r['SHAPValue']:<18.4f}")
        print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP value model explanations.")
    parser.add_argument('--index', type=int, default=None, help="Specific test dataframe row index to explain local prediction.")
    args = parser.parse_args()
    
    print("Loading resources...")
    lgb_model, preprocessor, X_test, y_test = load_resources()
    
    # 1. Global Explanations
    generate_global_explanations(lgb_model, X_test)
    
    # 2. Local Explanations
    explain_local_prediction(lgb_model, X_test, y_test, index=args.index)
