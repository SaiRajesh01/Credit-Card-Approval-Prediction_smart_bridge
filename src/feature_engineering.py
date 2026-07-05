import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FeatureEngineering")

# Add project root to path to load config
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import get_config
from src.data_preprocessing import DataLoader, DataCleaner

config = get_config()

class FeatureEngineer:
    """Class to perform feature engineering, label creation, and dataset merging."""

    def __init__(self) -> None:
        """Initialize paths and ensure output directory exists."""
        self.final_dataset_path = config.FINAL_DATASET_PATH
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def generate_labels(self, credit_df: pd.DataFrame) -> pd.DataFrame:
        """Creates target label from credit payment history status.
        
        Good payment (STATUS 0, X, C) -> Approved (1)
        Bad payment (STATUS 1, 2, 3, 4, 5) -> Rejected (0)
        
        Args:
            credit_df (pd.DataFrame): Raw credit records.
            
        Returns:
            pd.DataFrame: DataFrame containing ['ID', 'STATUS_LABEL'].
        """
        logger.info("Aggregating credit history records to create binary targets...")
        bad_status = {'1', '2', '3', '4', '5'}
        credit_df['is_delinquent'] = credit_df['STATUS'].apply(lambda x: 1 if x in bad_status else 0)
        aggregated = credit_df.groupby('ID')['is_delinquent'].sum().reset_index()
        aggregated['STATUS_LABEL'] = aggregated['is_delinquent'].apply(lambda x: 0 if x > 0 else 1)
        
        logger.info(f"Target distribution generated: "
                    f"Approved: {(aggregated['STATUS_LABEL'] == 1).sum()} ({100 * (aggregated['STATUS_LABEL'] == 1).mean():.2f}%), "
                    f"Rejected: {(aggregated['STATUS_LABEL'] == 0).sum()} ({100 * (aggregated['STATUS_LABEL'] == 0).mean():.2f}%)")
        return aggregated[['ID', 'STATUS_LABEL']]

    def engineer_features(self, app_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms days-based fields and engineers new features.
        
        Args:
            app_df (pd.DataFrame): Cleaned application records.
            
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        logger.info("Engineering features from DAYS_BIRTH and DAYS_EMPLOYED...")
        df_feats = app_df.copy()
        
        # 1. Transform DAYS_BIRTH -> AGE_YEARS
        df_feats['AGE_YEARS'] = (-df_feats['DAYS_BIRTH']) / 365.25
        df_feats['AGE_YEARS'] = df_feats['AGE_YEARS'].round(1)
        
        # 2. Transform DAYS_EMPLOYED
        df_feats['IS_UNEMPLOYED'] = (df_feats['DAYS_EMPLOYED'] == 365243).astype(int)
        df_feats['YEARS_EMPLOYED'] = df_feats['DAYS_EMPLOYED'].apply(
            lambda x: 0.0 if x == 365243 else (-x) / 365.25
        )
        df_feats['YEARS_EMPLOYED'] = df_feats['YEARS_EMPLOYED'].round(1)
        
        # 3. Extra engineered features
        df_feats['INCOME_PER_MEMBER'] = df_feats['AMT_INCOME_TOTAL'] / df_feats['CNT_FAM_MEMBERS']
        df_feats['EMPLOYED_TO_AGE_RATIO'] = df_feats['YEARS_EMPLOYED'] / df_feats['AGE_YEARS']
        df_feats['CNT_ADULTS'] = df_feats['CNT_FAM_MEMBERS'] - df_feats['CNT_CHILDREN']
        df_feats['INCOME_TO_AGE_RATIO'] = df_feats['AMT_INCOME_TOTAL'] / df_feats['AGE_YEARS']
        df_feats['INCOME_PER_ADULT'] = df_feats['AMT_INCOME_TOTAL'] / np.maximum(df_feats['CNT_ADULTS'], 1)
        df_feats['HAS_WORK_PHONE_AND_PHONE'] = (df_feats['FLAG_WORK_PHONE'] & df_feats['FLAG_PHONE']).astype(int)
        
        car_map = {'Y': 1, 'N': 0}
        realty_map = {'Y': 1, 'N': 0}
        car_val = df_feats['FLAG_OWN_CAR'].map(car_map)
        realty_val = df_feats['FLAG_OWN_REALTY'].map(realty_map)
        df_feats['ASSETS_COUNT'] = car_val + realty_val
        
        # Drop raw day count columns
        df_feats = df_feats.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED'])
        
        logger.info("Feature engineering completed successfully.")
        return df_feats

    def merge_and_save(self, app_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """Merges demographics with the target label on applicant ID and saves dataset.
        
        Args:
            app_df (pd.DataFrame): Engineered application records.
            target_df (pd.DataFrame): Aggregated labels dataframe.
            
        Returns:
            pd.DataFrame: Combined final dataset.
        """
        logger.info("Merging application profiles and target status labels on ID...")
        merged_df = pd.merge(app_df, target_df, on='ID', how='inner')
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        merged_df.to_csv(self.final_dataset_path, index=False)
        logger.info(f"Final merged dataset saved to: {self.final_dataset_path}")
        return merged_df


class FeaturePreprocessor:
    """Class to handle categorical encoding, feature scaling, and train-test splits."""

    def __init__(self) -> None:
        """Initialize Preprocessor and ensure the model artifacts directory exists."""
        self.models_dir = config.SAVED_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define feature groups
        self.categorical_cols = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
        ]
        self.numerical_cols = [
            'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AGE_YEARS', 
            'YEARS_EMPLOYED', 'CNT_FAM_MEMBERS',
            'INCOME_PER_MEMBER', 'EMPLOYED_TO_AGE_RATIO', 'CNT_ADULTS',
            'INCOME_TO_AGE_RATIO', 'INCOME_PER_ADULT', 'ASSETS_COUNT'
        ]
        # Binary features do not require encoding or scaling
        self.binary_cols = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'IS_UNEMPLOYED', 'HAS_WORK_PHONE_AND_PHONE']

    def process_and_split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Runs the preprocessing pipeline and splits data into Train/Test sets.
        
        1. Separates features X (dropping ID) and target y.
        2. Splits into train/test splits.
        3. Fits LabelEncoders on categorical columns and saves the encoders map.
        4. Fits StandardScaler on numerical columns and saves the scaler object.
        5. Saves feature columns for model alignment.
        
        Args:
            df (pd.DataFrame): Cleaned and merged final dataframe.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test.
        """
        logger.info("Preparing features and targets for split...")
        
        # Separate features and target
        X = df.drop(columns=['ID', 'STATUS_LABEL'])
        y = df['STATUS_LABEL']
        
        # Save feature column names list
        feature_columns = list(X.columns)
        joblib.dump(feature_columns, config.FEATURE_COLUMNS_PATH)
        logger.info(f"Feature columns metadata saved to: {config.FEATURE_COLUMNS_PATH}")
        
        # Train-Test Split (80-20, fixed random state for reproducibility)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        logger.info(f"Split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Encode Categorical Variables
        logger.info("Encoding categorical variables using LabelEncoder...")
        encoders: Dict[str, LabelEncoder] = {}
        
        # Fit encoders on the entire set first to capture all possible categories (safe for demographic metadata)
        # and transform train/test split.
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Fit on complete set
            le.fit(X[col])
            
            # Transform train and test
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            
            # Store fitted encoder
            encoders[col] = le
            
        # Serialize encoders dict
        joblib.dump(encoders, config.ENCODERS_PATH)
        logger.info(f"Serialized LabelEncoders saved to: {config.ENCODERS_PATH}")
        
        # Scale Numerical Variables
        logger.info("Scaling numerical variables using StandardScaler...")
        scaler = StandardScaler()
        
        # Fit ONLY on X_train to prevent data leakage
        X_train[self.numerical_cols] = scaler.fit_transform(X_train[self.numerical_cols])
        # Transform X_test
        X_test[self.numerical_cols] = scaler.transform(X_test[self.numerical_cols])
        
        # Serialize scaler
        joblib.dump(scaler, config.SCALER_PATH)
        logger.info(f"Serialized StandardScaler saved to: {config.SCALER_PATH}")
        
        # Convert to numpy arrays for modeling consistency
        return X_train.values, X_test.values, y_train.values, y_test.values


if __name__ == "__main__":
    try:
        # Load and clean data
        loader = DataLoader()
        raw_app, raw_credit = loader.load_data()
        
        cleaner = DataCleaner()
        clean_app = cleaner.clean_application_data(raw_app)
        clean_credit = cleaner.clean_credit_data(raw_credit)
        
        # Run Feature Engineering
        engineer = FeatureEngineer()
        target_df = engineer.generate_labels(clean_credit)
        engineered_app = engineer.engineer_features(clean_app)
        final_df = engineer.merge_and_save(engineered_app, target_df)
        
        # Run Encoding and Scaling Preprocessor
        preprocessor = FeaturePreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.process_and_split(final_df)
        
        logger.info("Encoding & Scaling complete. Dataset ready for model training.")
        logger.info(f"X_train sample features:\n{X_train[:1]}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        sys.exit(1)
