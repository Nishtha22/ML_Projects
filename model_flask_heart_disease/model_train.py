import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib
import json

def compute_class_weights(y):
    """Compute balanced class weights with extra boost for minority class"""
    classes = np.unique(y)
    weights = len(y) / (len(classes) * np.bincount(y))
    # Give extra weight to minority class
    weights[1] = weights[1] * 2
    return dict(zip(classes, weights))

def preprocess_data(df):
    """Preprocess the dataset"""
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                       'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                       'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level',
                       'Sugar Consumption']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    return df, label_encoders

def train_model(data_path='heart_disease.csv'):
    """Train the heart disease prediction model"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Preprocess data
    print("Preprocessing data...")
    df, label_encoders = preprocess_data(df)
    
    # Separate features and target
    target_col = 'Heart Disease Status'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with balanced class weights
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',  # Handle class imbalance
        min_samples_leaf=5  # Prevent overfitting to majority class
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model with balanced metrics
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report (with zero_division=1 to show actual precision):")
    print(classification_report(y_test, y_pred, 
                              target_names=target_encoder.classes_,
                              zero_division=1))  # Show actual precision values
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model and preprocessors
    print("\nSaving model and preprocessors...")
    joblib.dump(model, 'models/heart_disease_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(target_encoder, 'models/target_encoder.pkl')
    
    # Save feature names
    feature_names = X.columns.tolist()
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    print("Model training complete!")
    print(f"Model saved to: models/heart_disease_model.pkl")
    
    return model, scaler, label_encoders, target_encoder

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    train_model()