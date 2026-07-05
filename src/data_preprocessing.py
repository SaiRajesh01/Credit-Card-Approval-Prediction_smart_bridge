import logging
import sys
import urllib.request
from pathlib import Path
from typing import Tuple, List
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DataPreprocessing")

# Add project root to path to load config
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import get_config

config = get_config()

class DataLoader:
    """Class to handle loading and downloading of credit datasets."""
    
    APP_RECORD_URL = (
        "https://raw.githubusercontent.com/damaniayesh/Credit-Card-Approval-Prediction/master/application_record.csv"
    )
    CREDIT_RECORD_URL = (
        "https://raw.githubusercontent.com/damaniayesh/Credit-Card-Approval-Prediction/master/credit_record.csv"
    )

    def __init__(self) -> None:
        """Initialize DataLoader paths."""
        self.data_dir = config.DATA_DIR
        self.app_path = config.APPLICATION_RECORD_PATH
        self.credit_path = config.CREDIT_RECORD_PATH
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, url: str, destination_path: Path) -> None:
        """Downloads a dataset from a URL if it does not exist locally."""
        if destination_path.exists():
            logger.info(f"Dataset already exists locally at: {destination_path.name}")
            return
            
        logger.info(f"Downloading dataset from: {url} to {destination_path}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(destination_path, 'wb') as out_file:
                out_file.write(response.read())
            logger.info(f"Successfully downloaded and saved {destination_path.name}.")
        except Exception as e:
            logger.error(f"Failed to download dataset from {url}. Error: {e}")
            raise

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads both datasets into memory, downloading if necessary."""
        try:
            self.download_dataset(self.APP_RECORD_URL, self.app_path)
            self.download_dataset(self.CREDIT_RECORD_URL, self.credit_path)
            
            logger.info("Reading application_record.csv...")
            app_df = pd.read_csv(self.app_path)
            
            logger.info("Reading credit_record.csv...")
            credit_df = pd.read_csv(self.credit_path)
            
            return app_df, credit_df
        except Exception as e:
            logger.critical(f"Critical error loading data: {e}")
            raise


class DataCleaner:
    """Class to clean demographic and credit datasets."""

    def __init__(self, columns_to_drop: List[str] = ["FLAG_MOBIL"]) -> None:
        """Initialize cleaner with columns that should be dropped.
        
        Args:
            columns_to_drop (List[str]): Columns with zero variance or no utility.
        """
        self.columns_to_drop = columns_to_drop

    def clean_application_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans application records DataFrame.
        
        Handles:
        1. Dropping complete duplicate rows.
        2. Dropping duplicate IDs (retaining the first occurrence).
        3. Filling missing categorical values (OCCUPATION_TYPE) with 'Unknown'.
        4. Dropping zero-variance columns (like FLAG_MOBIL).
        
        Args:
            df (pd.DataFrame): Raw application records.
            
        Returns:
            pd.DataFrame: Cleaned application records.
        """
        logger.info("Starting cleaning pipeline for Application Data...")
        df_cleaned = df.copy()
        
        # 1. Drop exact duplicate rows
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        dropped_duplicates = initial_rows - len(df_cleaned)
        logger.info(f"Removed {dropped_duplicates} exact duplicate rows.")
        
        # 2. Handle duplicate IDs (ensure ID acts as unique primary key)
        initial_unique_ids = df_cleaned['ID'].nunique()
        initial_total_rows = len(df_cleaned)
        # Drop duplicates where ID matches, keeping first profile
        df_cleaned = df_cleaned.drop_duplicates(subset=['ID'], keep='first')
        dropped_id_duplicates = initial_total_rows - len(df_cleaned)
        logger.info(f"Removed {dropped_id_duplicates} duplicate ID rows. Unique IDs: {initial_unique_ids}")
        
        # 3. Impute missing OCCUPATION_TYPE values
        missing_occupation_count = df_cleaned['OCCUPATION_TYPE'].isnull().sum()
        df_cleaned['OCCUPATION_TYPE'] = df_cleaned['OCCUPATION_TYPE'].fillna("Unknown")
        logger.info(f"Imputed {missing_occupation_count} missing values in OCCUPATION_TYPE with 'Unknown'.")
        
        # 4. Drop columns with no predictive value/variance
        existing_cols_to_drop = [col for col in self.columns_to_drop if col in df_cleaned.columns]
        if existing_cols_to_drop:
            df_cleaned = df_cleaned.drop(columns=existing_cols_to_drop)
            logger.info(f"Dropped columns: {existing_cols_to_drop}")
            
        logger.info(f"Application Data cleaning complete. Final shape: {df_cleaned.shape}")
        return df_cleaned

    def clean_credit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans credit records DataFrame.
        
        Handles duplicate checks and removes invalid rows.
        
        Args:
            df (pd.DataFrame): Raw credit records.
            
        Returns:
            pd.DataFrame: Cleaned credit records.
        """
        logger.info("Starting cleaning pipeline for Credit Data...")
        df_cleaned = df.copy()
        
        # Drop duplicate credit records
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        dropped_duplicates = initial_rows - len(df_cleaned)
        logger.info(f"Removed {dropped_duplicates} duplicate credit status rows.")
        
        logger.info(f"Credit Data cleaning complete. Final shape: {df_cleaned.shape}")
        return df_cleaned


if __name__ == "__main__":
    try:
        loader = DataLoader()
        app_df, credit_df = loader.load_data()
        
        cleaner = DataCleaner()
        cleaned_app_df = cleaner.clean_application_data(app_df)
        cleaned_credit_df = cleaner.clean_credit_data(credit_df)
        
        logger.info(f"Cleaned App shape: {cleaned_app_df.shape}")
        logger.info(f"Cleaned Credit shape: {cleaned_credit_df.shape}")
        
    except Exception as e:
        logger.error(f"Error in data cleaning pipeline: {e}")
        sys.exit(1)
