import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Base Configuration class containing shared settings."""
    
    BASE_DIR: Path = BASE_DIR
    
    # Flask settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "default-secret-key-change-in-production")
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    
    # Database Settings
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'credit_card.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    
    # Application Paths
    DATA_DIR: Path = BASE_DIR / "data"
    NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
    SRC_DIR: Path = BASE_DIR / "src"
    SAVED_MODELS_DIR: Path = BASE_DIR / "saved_models"
    
    # Dataset Filenames
    APPLICATION_RECORD_PATH: Path = DATA_DIR / "application_record.csv"
    CREDIT_RECORD_PATH: Path = DATA_DIR / "credit_record.csv"
    FINAL_DATASET_PATH: Path = DATA_DIR / "final_dataset.csv"
    
    # Saved Artifact Paths
    BEST_MODEL_PATH: Path = SAVED_MODELS_DIR / "best_model.pkl"
    SCALER_PATH: Path = SAVED_MODELS_DIR / "scaler.pkl"
    ENCODERS_PATH: Path = SAVED_MODELS_DIR / "encoders.pkl"
    FEATURE_COLUMNS_PATH: Path = SAVED_MODELS_DIR / "feature_columns.pkl"
    
    # Model Hyperparameters Configuration
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    # Database configuration details
    DB_NAME: str = "credit_card.db"


class DevelopmentConfig(Config):
    """Development stage settings."""
    DEBUG: bool = True
    TESTING: bool = False


class ProductionConfig(Config):
    """Production stage settings."""
    DEBUG: bool = False
    TESTING: bool = False
    # In production, require a strong secret key
    SECRET_KEY = os.getenv("SECRET_KEY") or "prod-secret-must-be-set-in-env"


# Dictionary mapping configuration environments
config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}

def get_config() -> type[Config]:
    """Helper function to load the configuration class based on environment."""
    env = os.getenv("FLASK_ENV", "development").lower()
    return config_by_name.get(env, DevelopmentConfig)
