"""
LTV Forecasting Model for Mobile Games

This module provides functions to train and predict user Lifetime Value (LTV).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Feature columns used for modeling
FEATURE_COLS = [
    'acquisition_source_encoded', 'country_encoded', 'platform_encoded', 'age_group_encoded',
    'd1_retention', 'd7_retention', 'd30_retention',
    'total_sessions', 'avg_session_duration_min', 'total_playtime_hours',
    'levels_completed', 'tutorial_completed', 'ads_watched',
    'friends_invited', 'guild_joined',
    'engagement_score', 'sessions_per_day'
]

CATEGORICAL_COLS = ['acquisition_source', 'country', 'platform', 'age_group']


def prepare_features(df, label_encoders=None, fit=False):
    """
    Prepare features for LTV prediction.

    Args:
        df: DataFrame with user data
        label_encoders: Dict of fitted LabelEncoders (optional)
        fit: Whether to fit new encoders

    Returns:
        DataFrame with features, dict of label encoders
    """
    df_model = df.copy()

    if label_encoders is None:
        label_encoders = {}

    # Encode categorical variables
    for col in CATEGORICAL_COLS:
        if fit or col not in label_encoders:
            le = LabelEncoder()
            df_model[f'{col}_encoded'] = le.fit_transform(df_model[col])
            label_encoders[col] = le
        else:
            df_model[f'{col}_encoded'] = label_encoders[col].transform(df_model[col])

    # Create engagement score
    df_model['engagement_score'] = (
        df_model['d1_retention'] * 0.1 +
        df_model['d7_retention'] * 0.3 +
        df_model['d30_retention'] * 0.6 +
        np.log1p(df_model['total_sessions']) * 0.1 +
        df_model['tutorial_completed'] * 0.2
    )

    # Sessions per day
    df_model['sessions_per_day'] = df_model['total_sessions'] / np.maximum(df_model['days_since_install'], 1)

    return df_model, label_encoders


def train_ltv_model(df, target='ltv_day365', test_size=0.2):
    """
    Train an LTV prediction model.

    Args:
        df: DataFrame with user data
        target: Target column name
        test_size: Fraction of data to use for testing

    Returns:
        Trained model, scaler, label_encoders, metrics dict
    """
    # Prepare features
    df_model, label_encoders = prepare_features(df, fit=True)

    X = df_model[FEATURE_COLS]
    y = df_model[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, label_encoders, metrics, feature_importance


def predict_ltv(df, model, label_encoders):
    """
    Predict LTV for new users.

    Args:
        df: DataFrame with user data
        model: Trained model
        label_encoders: Fitted label encoders

    Returns:
        Array of LTV predictions
    """
    df_model, _ = prepare_features(df, label_encoders=label_encoders, fit=False)
    X = df_model[FEATURE_COLS]
    return model.predict(X)


def segment_users(ltv_values, percentiles=None):
    """
    Segment users based on predicted LTV.

    Args:
        ltv_values: Array of LTV values
        percentiles: Dict of percentile thresholds

    Returns:
        Array of segment labels
    """
    if percentiles is None:
        percentiles = {
            'p75': np.percentile(ltv_values[ltv_values > 0], 75),
            'p95': np.percentile(ltv_values[ltv_values > 0], 95)
        }

    segments = []
    for ltv in ltv_values:
        if ltv == 0:
            segments.append('Non-Payer')
        elif ltv < percentiles['p75']:
            segments.append('Low Value')
        elif ltv < percentiles['p95']:
            segments.append('Medium Value')
        else:
            segments.append('High Value (Whale)')

    return np.array(segments)


def save_model(model, label_encoders, output_dir):
    """Save model artifacts to disk."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'ltv_model.pkl'))
    joblib.dump(label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))


def load_model(model_dir):
    """Load model artifacts from disk."""
    model = joblib.load(os.path.join(model_dir, 'ltv_model.pkl'))
    label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
    return model, label_encoders


if __name__ == '__main__':
    # Load data
    print("Loading data...")
    df = pd.read_csv('/home/user/LTV-forecasting/data/mobile_game_users.csv', parse_dates=['install_date'])

    # Train model
    print("Training LTV model...")
    model, label_encoders, metrics, feature_importance = train_ltv_model(df)

    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"MAE: ${metrics['mae']:.2f}")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"R2 Score: {metrics['r2']:.4f}")

    print("\n" + "="*50)
    print("TOP FEATURES")
    print("="*50)
    print(feature_importance.head(10).to_string(index=False))

    # Save model
    save_model(model, label_encoders, '/home/user/LTV-forecasting/data')
    print("\nModel saved to /home/user/LTV-forecasting/data/")

    # Generate predictions
    print("\nGenerating predictions for all users...")
    predictions = predict_ltv(df, model, label_encoders)
    segments = segment_users(predictions)

    # Save predictions
    results = pd.DataFrame({
        'user_id': df['user_id'],
        'predicted_ltv': predictions.round(2),
        'segment': segments
    })
    results.to_csv('/home/user/LTV-forecasting/data/ltv_predictions.csv', index=False)

    print("\n" + "="*50)
    print("SEGMENT DISTRIBUTION")
    print("="*50)
    segment_summary = results.groupby('segment').agg({
        'user_id': 'count',
        'predicted_ltv': ['mean', 'sum']
    })
    segment_summary.columns = ['Count', 'Avg LTV', 'Total LTV']
    segment_summary['% of Users'] = (segment_summary['Count'] / len(results) * 100).round(1)
    print(segment_summary.to_string())

    print("\nPredictions saved to /home/user/LTV-forecasting/data/ltv_predictions.csv")
