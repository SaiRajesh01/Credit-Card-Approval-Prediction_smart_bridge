import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PredictionService")

# Add project root to path to load config
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import get_config
from src.data_preprocessing import DataLoader, DataCleaner
from src.feature_engineering import FeatureEngineer, FeaturePreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.ensemble import EnsembleClassifier

config = get_config()

class ModelSelector:
    """Class to compare trained models and serialize the best performing one."""

    def __init__(self) -> None:
        """Initialize the model selector."""
        self.best_model_path = config.BEST_MODEL_PATH

    def select_and_save(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, metric: str = "F1 Score") -> str:
        """Evaluates models, selects the best based on a metric, and serializes it.
        
        Args:
            models (Dict[str, Any]): Dictionary of trained models.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.
            metric (str): Metric to optimize (e.g., 'F1 Score' or 'ROC AUC').
            
        Returns:
            str: Name of the selected best model.
        """
        logger.info(f"Automatically selecting the best model based on: {metric}...")
        
        best_score = -1.0
        best_model_name = ""
        best_model_obj = None
        
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            if metric == "F1 Score":
                score = f1_score(y_test, y_pred)
            elif metric == "ROC AUC":
                score = roc_auc_score(y_test, y_prob)
            else:
                score = accuracy_score(y_test, y_pred)
                
            logger.info(f"Model: {name} | {metric}: {score:.4%}")
            
            if score > best_score:
                best_score = score
                best_model_name = name
                best_model_obj = model
                
        logger.info(f"Selected Best Model: {best_model_name} with {metric} = {best_score:.4%}")
        
        # Serialize best model
        joblib.dump(best_model_obj, self.best_model_path)
        logger.info(f"Serialized best model saved successfully to: {self.best_model_path}")
        
        return best_model_name


class CreditPredictor:
    """Class to perform predictions on raw applicant profiles."""

    def __init__(self) -> None:
        """Loads serialized model, label encoders, standard scaler, and feature columns metadata."""
        try:
            logger.info("Initializing CreditPredictor service...")
            self.model = joblib.load(config.BEST_MODEL_PATH)
            self.encoders = joblib.load(config.ENCODERS_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
            self.feature_columns = joblib.load(config.FEATURE_COLUMNS_PATH)
            logger.info("CreditPredictor service loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Missing serialization assets. Ensure Module 8 & 11 have executed. Error: {e}")
            raise RuntimeError("Incomplete serialization artifacts.")

    def preprocess_input(self, applicant_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocesses a raw dictionary of applicant details.
        
        Maps input fields to match engineered training columns:
        - Calculates AGE_YEARS from birthdate or directly from inputs.
        - Calculates YEARS_EMPLOYED and IS_UNEMPLOYED from employment status.
        - Encodes categorical columns.
        - Scales numerical columns.
        
        Args:
            applicant_data (Dict[str, Any]): Raw user inputs.
            
        Returns:
            pd.DataFrame: Preprocessed single-row DataFrame matching training layout.
        """
        # Convert raw applicant dict to DataFrame
        df_input = pd.DataFrame([applicant_data])
        
        # 1. Feature Engineering
        # Handle DAYS_BIRTH -> AGE_YEARS (If age is provided in years, convert to match age units)
        if 'DAYS_BIRTH' in df_input.columns:
            df_input['AGE_YEARS'] = (-df_input['DAYS_BIRTH']) / 365.25
        elif 'AGE_YEARS' not in df_input.columns:
            # Fallback default age
            df_input['AGE_YEARS'] = 35.0
            
        # Handle DAYS_EMPLOYED / YEARS_EMPLOYED & IS_UNEMPLOYED
        if 'DAYS_EMPLOYED' in df_input.columns:
            val = df_input.loc[0, 'DAYS_EMPLOYED']
            df_input['IS_UNEMPLOYED'] = 1 if val == 365243 else 0
            df_input['YEARS_EMPLOYED'] = 0.0 if val == 365243 else (-val) / 365.25
        else:
            if 'IS_UNEMPLOYED' not in df_input.columns:
                df_input['IS_UNEMPLOYED'] = 0
            if 'YEARS_EMPLOYED' not in df_input.columns:
                df_input['YEARS_EMPLOYED'] = 5.0
                
        # Extra engineered features
        df_input['INCOME_PER_MEMBER'] = df_input['AMT_INCOME_TOTAL'] / df_input['CNT_FAM_MEMBERS']
        df_input['EMPLOYED_TO_AGE_RATIO'] = df_input['YEARS_EMPLOYED'] / df_input['AGE_YEARS']
        df_input['CNT_ADULTS'] = df_input['CNT_FAM_MEMBERS'] - df_input['CNT_CHILDREN']
        df_input['INCOME_TO_AGE_RATIO'] = df_input['AMT_INCOME_TOTAL'] / df_input['AGE_YEARS']
        df_input['INCOME_PER_ADULT'] = df_input['AMT_INCOME_TOTAL'] / np.maximum(df_input['CNT_ADULTS'], 1.0)
        df_input['HAS_WORK_PHONE_AND_PHONE'] = (df_input['FLAG_WORK_PHONE'] & df_input['FLAG_PHONE']).astype(int)
        
        car_map = {'Y': 1, 'N': 0}
        realty_map = {'Y': 1, 'N': 0}
        car_val = df_input['FLAG_OWN_CAR'].map(car_map)
        realty_val = df_input['FLAG_OWN_REALTY'].map(realty_map)
        df_input['ASSETS_COUNT'] = car_val + realty_val
                
        # Drop original raw date columns if present
        cols_to_drop = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'ID', 'STATUS_LABEL', 'FLAG_MOBIL']
        existing_drops = [col for col in cols_to_drop if col in df_input.columns]
        df_input = df_input.drop(columns=existing_drops, errors='ignore')
        
        # 2. Categorical Encoding
        categorical_cols = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
        ]
        
        for col in categorical_cols:
            if col in df_input.columns:
                val = df_input.loc[0, col]
                le = self.encoders[col]
                # Handle unseen categories gracefully in production
                if val not in le.classes_:
                    logger.warning(f"Unseen category '{val}' for column '{col}'. Defaulting to first class.")
                    df_input.loc[0, col] = le.classes_[0]
                df_input[col] = le.transform(df_input[col])
                
        # 3. Numerical Scaling
        numerical_cols = [
            'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AGE_YEARS', 
            'YEARS_EMPLOYED', 'CNT_FAM_MEMBERS',
            'INCOME_PER_MEMBER', 'EMPLOYED_TO_AGE_RATIO', 'CNT_ADULTS',
            'INCOME_TO_AGE_RATIO', 'INCOME_PER_ADULT', 'ASSETS_COUNT'
        ]
        df_input[numerical_cols] = self.scaler.transform(df_input[numerical_cols])
        
        # Align features ordering exactly with the training signature
        df_input = df_input[self.feature_columns]
        return df_input

    def predict_approval(self, applicant_data: Dict[str, Any]) -> Tuple[int, float]:
        """Predicts credit card approval.
        
        Args:
            applicant_data (Dict[str, Any]): Raw user profile metadata.
            
        Returns:
            Tuple[int, float]: (STATUS_LABEL [1=Approved, 0=Rejected], confidence score [0.0 - 1.0]).
        """
        # Preprocess dict to scaled feature vector
        processed_df = self.preprocess_input(applicant_data)
        
        # Predict class label
        prediction = int(self.model.predict(processed_df.values)[0])
        
        # Predict probabilities
        probabilities = self.model.predict_proba(processed_df.values)[0]
        confidence = float(probabilities[prediction])
        
        logger.info(f"Prediction: {prediction} | Confidence: {confidence:.2%}")
        return prediction, confidence


if __name__ == "__main__":
    try:
        # Load preprocessed datasets
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
        
        # Train classifiers
        trainer = ModelTrainer()
        trained_models = trainer.train_all(X_train, y_train)
        
        # Compare and Save the Best Model
        selector = ModelSelector()
        best_name = selector.select_and_save(trained_models, X_test, y_test, metric="ROC AUC")
        
        # Test real-time prediction facade
        predictor = CreditPredictor()
        
        # Create a mock applicant profile for prediction testing
        mock_applicant = {
            'CODE_GENDER': 'M',
            'FLAG_OWN_CAR': 'Y',
            'FLAG_OWN_REALTY': 'Y',
            'CNT_CHILDREN': 0,
            'AMT_INCOME_TOTAL': 200000.0,
            'NAME_INCOME_TYPE': 'Working',
            'NAME_EDUCATION_TYPE': 'Higher education',
            'NAME_FAMILY_STATUS': 'Married',
            'NAME_HOUSING_TYPE': 'House / apartment',
            'DAYS_BIRTH': -12000,      # ~32.8 years old
            'DAYS_EMPLOYED': -2000,    # ~5.5 years employed
            'FLAG_WORK_PHONE': 1,
            'FLAG_PHONE': 0,
            'FLAG_EMAIL': 1,
            'OCCUPATION_TYPE': 'Managers',
            'CNT_FAM_MEMBERS': 2.0
        }
        
        prediction, confidence = predictor.predict_approval(mock_applicant)
        result_text = "APPROVED" if prediction == 1 else "REJECTED"
        logger.info(f"Self-Test prediction for applicant: {result_text} (Confidence: {confidence:.2%})")
        
    except Exception as e:
        logger.error(f"Error in prediction service pipeline: {e}")
        sys.exit(1)
