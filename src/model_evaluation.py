import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ModelEvaluation")

# Add project root to path to load config and training modules
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import get_config
from src.data_preprocessing import DataLoader, DataCleaner
from src.feature_engineering import FeatureEngineer, FeaturePreprocessor
from src.model_training import ModelTrainer

config = get_config()

class ModelEvaluator:
    """Class to evaluate and compare trained machine learning models."""

    def __init__(self) -> None:
        """Initialize evaluator and ensure visual output path exists."""
        self.image_dir = config.BASE_DIR / "static" / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # Set evaluation plot parameters
        sns.set_theme(style="whitegrid", context="talk")

    def evaluate_all(
        self, 
        models: Dict[str, Any], 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """Computes evaluation metrics, confusion matrices, and prints reports.
        
        Args:
            models (Dict[str, Any]): Dictionary of trained model objects.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.
            
        Returns:
            pd.DataFrame: Table comparing the metrics of all models.
        """
        logger.info("Starting model evaluation pipeline...")
        comparison_data = []
        
        # Prepare figure for ROC Curves
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Predict labels
            y_pred = model.predict(X_test)
            
            # Predict probabilities for class 1 (Approved)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            # Add metrics to comparison list
            comparison_data.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": roc_auc
            })
            
            # Display detailed text diagnostics
            print("\n" + "=" * 60)
            print(f" {name.upper()} CLASSIFICATION METRICS ".center(60, "#"))
            print("=" * 60)
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))
            
            print("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print(f"True Negatives (Rejections Correct): {cm[0][0]}")
            print(f"False Positives (Rejections Wrong): {cm[0][1]}")
            print(f"False Negatives (Approvals Wrong): {cm[1][0]}")
            print(f"True Positives (Approvals Correct): {cm[1][1]}")
            print("=" * 60 + "\n")
            
            # Compute ROC Curve details
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", linewidth=2.5)

        # Plot final ROC curves details
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier", alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison")
        plt.legend(loc="lower right")
        
        roc_plot_path = self.image_dir / "roc_curves.png"
        plt.savefig(roc_plot_path, dpi=150)
        plt.close()
        logger.info(f"Combined ROC curve chart saved to: {roc_plot_path}")
        
        # Build comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "=" * 85)
        print(f" MODEL COMPARISON TABLE ".center(85, "#"))
        print("=" * 85)
        print(comparison_df.to_string(index=False, formatters={
            "Accuracy": "{:.4%}".format,
            "Precision": "{:.4%}".format,
            "Recall": "{:.4%}".format,
            "F1 Score": "{:.4%}".format,
            "ROC AUC": "{:.4%}".format
        }))
        print("=" * 85 + "\n")
        
        return comparison_df


if __name__ == "__main__":
    try:
        # Pipeline execution
        loader = DataLoader()
        raw_app, raw_credit = loader.load_data()
        
        cleaner = DataCleaner()
        clean_app = cleaner.clean_application_data(raw_app)
        clean_credit = cleaner.clean_credit_data(raw_credit)
        
        engineer = FeatureEngineer()
        target_df = engineer.generate_labels(clean_credit)
        engineered_app = engineer.engineer_features(clean_app)
        final_df = engineer.merge_and_save(engineered_app, target_df)
        
        preprocessor = FeaturePreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.process_and_split(final_df)
        
        # Train models
        trainer = ModelTrainer()
        trained_models = trainer.train_all(X_train, y_train)
        
        # Evaluate models
        evaluator = ModelEvaluator()
        comparison_table = evaluator.evaluate_all(trained_models, X_test, y_test)
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline execution: {e}")
        sys.exit(1)
