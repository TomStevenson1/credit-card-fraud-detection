import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    """Neural Network for credit card fraud detection"""
    
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.sigmoid(self.output_layer(x))
        return x


class LogisticRegressionModel(nn.Module):
    """PyTorch Logistic Regression Model"""
    
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def load_and_preprocess_data(data_path):
    """Load and preprocess the credit card fraud dataset"""
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    print(f"Data shape: {data.shape}")
    print(f"Fraud cases: {data['Class'].sum()}")
    print(f"Non-fraud cases: {len(data) - data['Class'].sum()}")
    
    # Drop Time column and separate features from target
    X = data.drop(columns=['Class', 'Time'])
    y = data['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_sgd_model(X_train, y_train, X_test, y_test):
    """Train SGD Classifier (Linear SVM)"""
    print("Training SGD Classifier (Linear SVM)...")
    
    model = SGDClassifier(
        loss='hinge',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.decision_function(X_test)
    
    return model, y_pred, y_pred_proba


def train_logistic_regression_sklearn(X_train, y_train, X_test, y_test):
    """Train Logistic Regression using sklearn"""
    print("Training Logistic Regression (sklearn)...")
    
    model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba


def train_logistic_regression_pytorch(X_train, y_train, X_test, y_test):
    """Train Logistic Regression using PyTorch"""
    print("Training Logistic Regression (PyTorch)...")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = LogisticRegressionModel(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # Predictions
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).squeeze().numpy()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return model, y_pred, y_pred_proba


def train_neural_network(X_train, y_train, X_test, y_test):
    """Train Neural Network using PyTorch"""
    print("Training Neural Network...")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = NeuralNetwork(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Predictions
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).squeeze().numpy()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return model, y_pred, y_pred_proba


def evaluate_model(y_test, y_pred, y_pred_proba, model_name):
    """Evaluate model performance"""
    print(f"\n=== {model_name} Evaluation ===")
    
    # Basic metrics
    accuracy = (y_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate precision, recall, f1 manually from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score_val = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0.0
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score_val,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return metrics


def save_model_and_artifacts(model, scaler, metrics, model_name, output_dir):
    """Save model, scaler, and metrics"""
    # Create output directories
    models_dir = Path(output_dir) / "models"
    artifacts_dir = Path(output_dir) / "artifacts"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_dir / f"{model_name}_model.pkl"
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), model_path)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save scaler
    scaler_path = models_dir / f"{model_name}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics
    metrics_path = artifacts_dir / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Train credit card fraud detection models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['sgd', 'logreg', 'logreg_pytorch', 'nn'],
                       help='Model to train: sgd, logreg, logreg_pytorch, nn')
    parser.add_argument('--data', type=str, default='data/creditcard.csv',
                       help='Path to credit card dataset')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for models and artifacts')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(args.data)
    
    # Train model based on argument
    if args.model == 'sgd':
        model, y_pred, y_pred_proba = train_sgd_model(X_train, y_train, X_test, y_test)
        model_name = 'sgd'
    elif args.model == 'logreg':
        model, y_pred, y_pred_proba = train_logistic_regression_sklearn(X_train, y_train, X_test, y_test)
        model_name = 'logreg'
    elif args.model == 'logreg_pytorch':
        model, y_pred, y_pred_proba = train_logistic_regression_pytorch(X_train, y_train, X_test, y_test)
        model_name = 'logreg_pytorch'
    elif args.model == 'nn':
        model, y_pred, y_pred_proba = train_neural_network(X_train, y_train, X_test, y_test)
        model_name = 'nn'
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name)
    
    # Save model and artifacts
    save_model_and_artifacts(model, scaler, metrics, model_name, args.output)
    
    print(f"\n✅ Training completed successfully!")
    print(f"Model: {model_name}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()