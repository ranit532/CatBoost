
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_model():
    """
    Trains a CatBoost model on the diabetes dataset and logs the experiment with MLflow.
    """
    # Create directories if they don't exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load the dataset
    try:
        data = pd.read_csv("diabetes_dataset.csv")
    except FileNotFoundError:
        print("Error: 'diabetes_dataset.csv' not found. Please run generate_data.py first.")
        return

    # Split data into features and target
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set the MLflow experiment
    mlflow.set_experiment("Diabetes Prediction - CatBoost")

    # Start an MLflow run
    with mlflow.start_run():
        # Define model parameters
        params = {
            'iterations': 200,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'Logloss',
            'verbose': False
        }

        # Initialize and train the CatBoost model
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # --- Feature Importance Plot ---
        feature_importance = model.get_feature_importance(prettified=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importances", y="Feature Id", data=feature_importance)
        plt.title("CatBoost Feature Importance")
        feature_importance_path = "images/feature_importance.png"
        plt.savefig(feature_importance_path)
        plt.close()
        mlflow.log_artifact(feature_importance_path, "plots")

        # --- Confusion Matrix Plot ---
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        confusion_matrix_path = "images/confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()
        mlflow.log_artifact(confusion_matrix_path, "plots")

        # Log the model to MLflow
        mlflow.catboost.log_model(model, "catboost_model")

        # Save the model for FastAPI serving
        model.save_model("models/catboost_model.bin")

        print("\nModel training and logging complete.")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("Check the 'mlruns' directory or MLflow UI for more details.")

if __name__ == "__main__":
    train_model()
