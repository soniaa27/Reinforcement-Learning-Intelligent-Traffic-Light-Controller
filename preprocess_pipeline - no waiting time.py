
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

def preprocess_traffic_data(input_file='smart_traffic_management_dataset.csv'):
    print("="*80)
    print("TRAFFIC DATA PREPROCESSING PIPELINE")
    print("="*80)

    # Load dataset
    df = pd.read_csv(input_file)
    print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Temporal processing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | 
                          (df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)

    # Feature engineering
    df['queue_length'] = df['traffic_volume'] / (df['avg_vehicle_speed'] + 1e-5)
    df['vehicle_density'] = (df['vehicle_count_cars'] + 
                             df['vehicle_count_trucks'] + 
                             df['vehicle_count_bikes'])
    df['congestion_index'] = ((df['traffic_volume'] / df['traffic_volume'].max()) * 
                              (1 / (df['avg_vehicle_speed'] / df['avg_vehicle_speed'].max())))
    df['heavy_vehicle_ratio'] = df['vehicle_count_trucks'] / (df['vehicle_density'] + 1e-5)
    df['bike_ratio'] = df['vehicle_count_bikes'] / (df['vehicle_density'] + 1e-5)
    df['weather_impact'] = (df['humidity'] / 100.0) * (df['temperature'] / df['temperature'].max())

    print("\nFeatures engineered: queue_length, vehicle_density, congestion_index, etc.")

    # One-hot encoding
    weather_dummies = pd.get_dummies(df['weather_condition'], prefix='weather')
    signal_dummies = pd.get_dummies(df['signal_status'], prefix='signal')
    df = pd.concat([df, weather_dummies, signal_dummies], axis=1)

    print(f"Categorical encoding: {len(weather_dummies.columns)} weather + {len(signal_dummies.columns)} signal categories")

    # Define features
    numerical_features = [
        'location_id', 'traffic_volume', 'avg_vehicle_speed', 
        'vehicle_count_cars', 'vehicle_count_trucks', 'vehicle_count_bikes',
        'temperature', 'humidity', 'hour', 'minute', 'day_of_week',
        'is_peak_hour', 'queue_length', 'vehicle_density', 'congestion_index',
        'heavy_vehicle_ratio', 'bike_ratio', 'weather_impact'
    ]

    categorical_features = list(weather_dummies.columns) + list(signal_dummies.columns)
    state_columns = numerical_features + categorical_features

    # Normalization
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    print(f"\nNormalization applied: {len(numerical_features)} numerical features scaled to [0,1]")

    # Create state vectors
    X = df[state_columns].values
    y_accident = df['accident_reported'].values
    y_signal = df['signal_status'].values

    print(f"\nState vectors created: {X.shape[1]}-dimensional state space")

    # Split: 70% train, 15% val, 15% test
    indices = np.arange(len(df))
    idx_temp, idx_test = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.176, random_state=42, shuffle=True)

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_accident_train = y_accident[idx_train]
    y_accident_val = y_accident[idx_val]
    y_accident_test = y_accident[idx_test]
    y_signal_train = y_signal[idx_train]
    y_signal_val = y_signal[idx_val]
    y_signal_test = y_signal[idx_test]

    # Get dataframe splits for CSV export
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    print(f"\nDataset split:")
    print(f"  Train: {X_train.shape[0]} samples (70%)")
    print(f"  Validation: {X_val.shape[0]} samples (15%)")
    print(f"  Test: {X_test.shape[0]} samples (15%)")

    # Save Part 1: Training data
    train_data = {
        'X_train': X_train,
        'y_accident_train': y_accident_train,
        'y_signal_train': y_signal_train,
        'feature_names': state_columns,
        'scaler': scaler,
        'state_dim': X.shape[1]
    }

    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)

    df_train.to_csv('train_data.csv', index=False)

    print("\n[Part 1] Training data saved:")
    print("  - train_data.pkl (pickle format)")
    print("  - train_data.csv (CSV format)")

    # Save Part 2: Validation data
    val_data = {
        'X_val': X_val,
        'y_accident_val': y_accident_val,
        'y_signal_val': y_signal_val,
        'feature_names': state_columns,
        'state_dim': X.shape[1]
    }

    with open('val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)

    df_val.to_csv('val_data.csv', index=False)

    print("[Part 2] Validation data saved:")
    print("  - val_data.pkl (pickle format)")
    print("  - val_data.csv (CSV format)")

    # Save Part 3: Test data
    test_data = {
        'X_test': X_test,
        'y_accident_test': y_accident_test,
        'y_signal_test': y_signal_test,
        'feature_names': state_columns,
        'state_dim': X.shape[1]
    }

    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    df_test.to_csv('test_data.csv', index=False)

    print("[Part 3] Test data saved:")
    print("  - test_data.pkl (pickle format)")
    print("  - test_data.csv (CSV format)")

    # Save scaler separately
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\nScaler saved: scaler.pkl")

    # Save full preprocessed dataframe
    df.to_csv('full_preprocessed_data.csv', index=False)
    print("Full dataset saved: full_preprocessed_data.csv")

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  PICKLE FORMAT (for RL training):")
    print("    1. train_data.pkl")
    print("    2. val_data.pkl")
    print("    3. test_data.pkl")
    print("    4. scaler.pkl")
    print("\n  CSV FORMAT (for analysis/SUMO):")
    print("    1. train_data.csv")
    print("    2. val_data.csv")
    print("    3. test_data.csv")
    print("    4. full_preprocessed_data.csv")
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    preprocess_traffic_data()
