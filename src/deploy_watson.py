import logging
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("WatsonDeployment")

# Add project root to path to load config
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import get_config

config = get_config()

try:
    from ibm_watsonx_ai import APIClient 
except ImportError:
    logger.warning("ibm-watsonx-ai package is not installed. Run 'pip install ibm-watsonx-ai' to execute this script.")

class WatsonDeployer:
    """Class to automate model deployment to IBM Watson Machine Learning (WML)."""

    def __init__(self) -> None:
        """Initialize deployer, retrieving cloud credentials from the environment."""
        self.api_key = os.getenv("WATSON_API_KEY")
        self.wml_url = os.getenv("WATSON_URL", "https://us-south.ml.cloud.ibm.com")
        self.space_id = os.getenv("WATSON_SPACE_ID")
        
        # Paths
        self.model_path = config.BEST_MODEL_PATH
        self.tar_path = config.SAVED_MODELS_DIR / "best_model.tar.gz"

    def compress_model(self) -> Path:
        """Compresses the best_model.pkl file into a tar.gz archive for WML requirements.
        
        Returns:
            Path: Tarball archive path.
        """
        logger.info(f"Compressing {self.model_path.name} to tarball...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Champion model not found at {self.model_path}. Run training first.")
            
        with tarfile.open(self.tar_path, "w:gz") as tar:
            tar.add(self.model_path, arcname=self.model_path.name)
            
        logger.info(f"Model compressed successfully to: {self.tar_path}")
        return self.tar_path

    def deploy(self) -> str:
        """Connects to IBM WML client, uploads the tarball, and creates an online deployment.
        
        Returns:
            str: REST scoring URL for the WML online endpoint.
        """
        if not self.api_key or not self.space_id:
            raise ValueError("WATSON_API_KEY and WATSON_SPACE_ID environment variables must be defined.")
            
        # Compress model first
        tarball_path = self.compress_model()
        
        # 1. Initialize API Client
        logger.info("Connecting to IBM Watson Machine Learning client...")
        wml_credentials = {
            "url": self.wml_url,
            "apikey": self.api_key
        }
        client = APIClient(wml_credentials)
        
        # Set target deployment space
        client.set.default_space(self.space_id)
        logger.info(f"Deployment space set successfully: {self.space_id}")
        
        # 2. Configure software and model specifications
        # Retrieve software specification ID for XGBoost / Scikit-learn
        # WML default specifications: runtime-22.1-py3.9 / default_py3.9
        software_spec_uid = client.software_specifications.get_id_by_name("runtime-22.1-py3.9")
        logger.info(f"Retrieved software specification ID: {software_spec_uid}")
        
        model_props = {
            client.repository.ModelMetaNames.NAME: "Credit Card Approval Prediction XGBoost",
            client.repository.ModelMetaNames.TYPE: "xgboost_1.6",
            client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid
        }
        
        # 3. Store model in Watson Repository
        logger.info("Uploading model tarball to IBM Cloud repository...")
        model_details = client.repository.store_model(
            model=str(tarball_path),
            meta_props=model_props
        )
        model_uid = client.repository.get_model_id(model_details)
        logger.info(f"Model uploaded successfully. WML Model ID: {model_uid}")
        
        # 4. Create online REST deployment
        logger.info("Creating online deployment endpoint...")
        deployment_props = {
            client.deployments.ConfigurationMetaNames.NAME: "Credit_Card_Approval_Online_Endpoint",
            client.deployments.ConfigurationMetaNames.ONLINE: {}
        }
        
        deployment_details = client.deployments.create(
            artifact_uid=model_uid,
            meta_props=deployment_props
        )
        
        scoring_url = client.deployments.get_scoring_href(deployment_details)
        logger.info(f"WML deployment complete. REST Scoring URL: {scoring_url}")
        
        # Clean up local compressed archive
        if tarball_path.exists():
            tarball_path.unlink()
            logger.info("Cleaned up temporary tarball file.")
            
        return scoring_url


if __name__ == "__main__":
    # Check credentials
    if not os.getenv("WATSON_API_KEY") or not os.getenv("WATSON_SPACE_ID"):
        logger.error("Missing environment variables. Make sure WATSON_API_KEY and WATSON_SPACE_ID are set.")
        logger.info("Usage: Run with credentials set in terminal or .env file.")
        sys.exit(1)
        
    try:
        deployer = WatsonDeployer()
        scoring_endpoint = deployer.deploy()
        print(f"\n[DEPLOYMENT SUCCESS] Scoring Endpoint: {scoring_endpoint}\n")
    except Exception as e:
        logger.error(f"Watson deployment pipeline failed: {e}")
        sys.exit(1)
