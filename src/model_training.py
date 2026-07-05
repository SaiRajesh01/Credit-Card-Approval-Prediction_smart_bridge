import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ModelTraining")

# Add project root to path to load config and preprocessing modules
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import get_config
from src.data_preprocessing import DataLoader, DataCleaner
from src.feature_engineering import FeatureEngineer, FeaturePreprocessor
from src.ensemble import EnsembleClassifier

config = get_config()

class ModelTrainer:
    """Class to train multiple classification models for credit approval prediction."""

    def __init__(self) -> None:
        """Initialize trainer parameters."""
        self.random_state = config.RANDOM_STATE
        self.models: Dict[str, Any] = {}

    def get_imbalance_ratio(self, y_train: np.ndarray) -> float:
        """Calculates the ratio of negative (Rejected) to positive (Approved) class samples.
        
        This ratio is used to scale weights for XGBoost scale_pos_weight.
        
        Args:
            y_train (np.ndarray): Training target values (1=Approved, 0=Rejected).
            
        Returns:
            float: Imbalance ratio.
        """
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        # To balance: scale_pos_weight = count(negative) / count(positive)
        # Note: if positive is Approved (majority) and negative is Rejected (minority):
        # class_weight='balanced' in sklearn handles both automatically.
        # For XGBoost, scale_pos_weight is typically sum(negative) / sum(positive).
        # We calculate it as negative_count / positive_count.
        ratio = neg_count / pos_count if pos_count > 0 else 1.0
        return ratio

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Trains Logistic Regression, Decision Tree, Random Forest, XGBoost, and Ensemble models.
        
        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training target array.
            
        Returns:
            Dict[str, Any]: Dictionary of trained model names mapped to model objects.
        """
        logger.info("Initializing model training pipeline...")
        imbalance_ratio = self.get_imbalance_ratio(y_train)
        logger.info(f"Class imbalance ratio (Rejected/Approved): {imbalance_ratio:.4f}")

        # 1. Logistic Regression
        logger.info("Training Logistic Regression model...")
        lr = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight="balanced"
        )
        lr.fit(X_train, y_train)
        self.models["Logistic Regression"] = lr

        # 2. Decision Tree
        logger.info("Training Decision Tree Classifier...")
        dt = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=10,
            class_weight="balanced"
        )
        dt.fit(X_train, y_train)
        self.models["Decision Tree"] = dt

        # 3. Random Forest
        logger.info("Training Tuned Random Forest Classifier...")
        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models["Random Forest"] = rf

        # 4. XGBoost
        logger.info("Training Tuned and Balanced XGBoost Classifier...")
        xgb = XGBClassifier(
            random_state=self.random_state,
            n_estimators=300,
            learning_rate=0.2,
            max_depth=10,
            eval_metric="logloss",
            n_jobs=-1
        )
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        xgb.fit(X_train, y_train, sample_weight=sample_weights)
        self.models["XGBoost"] = xgb

        # 5. Ensemble (RF + XGB)
        logger.info("Building Ensemble Classifier...")
        ensemble = EnsembleClassifier(estimators=[rf, xgb])
        self.models["Ensemble (RF + XGB)"] = ensemble

        logger.info("All models trained successfully.")
        return self.models


if __name__ == "__main__":
    try:
        # Pipeline: Load -> Clean -> Engineer -> Split/Scale
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
        
        # Train
        trainer = ModelTrainer()
        trained_models = trainer.train_all(X_train, y_train)
        
        for name, model in trained_models.items():
            logger.info(f"Model: {name} successfully initialized and fitted.")
            
    except Exception as e:
        logger.error(f"Error in model training execution: {e}")
        sys.exit(1)
