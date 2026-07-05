import logging
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EDA")

# Add project root to path to load config
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import get_config

config = get_config()

class ExploratoryDataAnalysis:
    """Class to perform professional Exploratory Data Analysis and generate visualizations."""

    def __init__(self) -> None:
        """Initialize the EDA module and ensure the images output directory exists."""
        self.image_dir = config.BASE_DIR / "static" / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams.update({
            "figure.autolayout": True,
            "figure.titlesize": 20,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12
        })
        self.palette = sns.color_palette("coolwarm", as_cmap=False)
        self.primary_color = "#1f77b4"
        self.secondary_color = "#aec7e8"

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads raw datasets from data paths.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Application and Credit dataframes.
        """
        try:
            logger.info("Loading datasets...")
            app_df = pd.read_csv(config.APPLICATION_RECORD_PATH)
            credit_df = pd.read_csv(config.CREDIT_RECORD_PATH)
            return app_df, credit_df
        except FileNotFoundError as e:
            logger.error(f"Dataset files not found. Run Module 4 first. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise

    def process_labels(self, credit_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates and maps credit records to binary labels.
        
        Good Payment (0, X, C) -> Approved (1)
        Bad Payment (1, 2, 3, 4, 5) -> Rejected (0)
        
        Args:
            credit_df (pd.DataFrame): The raw credit history records.
            
        Returns:
            pd.DataFrame: Aggregated labels with ID and STATUS.
        """
        logger.info("Converting STATUS to binary classes...")
        # Map STATUS to binary (1 = Approved, 0 = Rejected)
        # Note: '1'-'5' are bad status (Rejected), '0', 'X', 'C' are good status (Approved)
        bad_status = {'1', '2', '3', '4', '5'}
        
        # Check if an applicant has any bad status in their history
        credit_df['is_bad'] = credit_df['STATUS'].apply(lambda x: 1 if x in bad_status else 0)
        
        # Group by ID: if sum of is_bad > 0, then user is Rejected (0), else Approved (1)
        grouped = credit_df.groupby('ID')['is_bad'].sum().reset_index()
        grouped['STATUS_LABEL'] = grouped['is_bad'].apply(lambda x: 0 if x > 0 else 1)
        
        logger.info(f"Target distribution after mapping:\n{grouped['STATUS_LABEL'].value_counts(normalize=True)}")
        return grouped[['ID', 'STATUS_LABEL']]

    def run_visualization_pipeline(self) -> None:
        """Runs the entire EDA pipeline and saves plots to disk."""
        app_df, credit_df = self.load_data()
        labels_df = self.process_labels(credit_df)
        
        # Merge datasets for combined analyses
        merged_df = pd.merge(app_df, labels_df, on='ID', how='inner')
        logger.info(f"Merged dataset shape for EDA: {merged_df.shape}")
        
        # 1. Income Distribution (Histogram with KDE)
        self.plot_income_distribution(merged_df)
        
        # 2. Occupation Distribution (Countplot)
        self.plot_occupation_distribution(merged_df)
        
        # 3. Education Distribution (Pie Chart & Countplot)
        self.plot_education_distribution(merged_df)
        
        # 4. Housing Type Distribution (Countplot)
        self.plot_housing_distribution(merged_df)
        
        # 5. Family Status (Countplot)
        self.plot_family_status_distribution(merged_df)
        
        # 6. Credit Status Distribution (Pie Chart)
        self.plot_credit_status_pie(labels_df)
        
        # 7. Boxplots (Income by Credit Status & Age by Credit Status)
        self.plot_demographic_boxplots(merged_df)
        
        # 8. Correlation Heatmap
        self.plot_correlation_heatmap(merged_df)

        logger.info("EDA Visualization Pipeline finished. All plots saved to static/images/")

    def plot_income_distribution(self, df: pd.DataFrame) -> None:
        """Plots the income distribution of applicants."""
        logger.info("Plotting Income Distribution...")
        plt.figure(figsize=(10, 6))
        
        # Filter out extreme income outliers for visualization clarity (e.g. 99th percentile)
        upper_limit = df['AMT_INCOME_TOTAL'].quantile(0.99)
        filtered_income = df[df['AMT_INCOME_TOTAL'] <= upper_limit]['AMT_INCOME_TOTAL']
        
        sns.histplot(filtered_income, kde=True, color="teal", bins=30)
        plt.title("Distribution of Annual Income (Filtered Outliers < 99th Pct)")
        plt.xlabel("Annual Income (USD)")
        plt.ylabel("Count")
        plt.savefig(self.image_dir / "income_distribution.png", dpi=150)
        plt.close()

    def plot_occupation_distribution(self, df: pd.DataFrame) -> None:
        """Plots the occupation type distribution."""
        logger.info("Plotting Occupation Distribution...")
        plt.figure(figsize=(12, 8))
        
        # Handle missing occupation type for EDA plot
        occ_df = df.copy()
        occ_df['OCCUPATION_TYPE'] = occ_df['OCCUPATION_TYPE'].fillna("Unspecified")
        
        order = occ_df['OCCUPATION_TYPE'].value_counts().index
        sns.countplot(data=occ_df, y='OCCUPATION_TYPE', order=order, palette="viridis")
        plt.title("Distribution of Applicant Occupations")
        plt.xlabel("Count")
        plt.ylabel("Occupation Type")
        plt.savefig(self.image_dir / "occupation_distribution.png", dpi=150)
        plt.close()

    def plot_education_distribution(self, df: pd.DataFrame) -> None:
        """Plots education level distributions using a Countplot and a Pie Chart."""
        logger.info("Plotting Education Distribution...")
        
        # Set up a 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Countplot
        order = df['NAME_EDUCATION_TYPE'].value_counts().index
        sns.countplot(data=df, y='NAME_EDUCATION_TYPE', order=order, ax=axes[0], palette="Blues_r")
        axes[0].set_title("Education Level Counts")
        axes[0].set_xlabel("Count")
        axes[0].set_ylabel("Education Type")
        
        # Pie Chart
        edu_counts = df['NAME_EDUCATION_TYPE'].value_counts()
        axes[1].pie(
            edu_counts, 
            labels=edu_counts.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette("pastel")
        )
        axes[1].set_title("Education Level Share")
        
        plt.savefig(self.image_dir / "education_distribution.png", dpi=150)
        plt.close()

    def plot_housing_distribution(self, df: pd.DataFrame) -> None:
        """Plots housing type distribution."""
        logger.info("Plotting Housing Distribution...")
        plt.figure(figsize=(10, 6))
        order = df['NAME_HOUSING_TYPE'].value_counts().index
        sns.countplot(data=df, y='NAME_HOUSING_TYPE', order=order, palette="Set2")
        plt.title("Distribution of Housing Types")
        plt.xlabel("Count")
        plt.ylabel("Housing Type")
        plt.savefig(self.image_dir / "housing_distribution.png", dpi=150)
        plt.close()

    def plot_family_status_distribution(self, df: pd.DataFrame) -> None:
        """Plots family status distribution."""
        logger.info("Plotting Family Status Distribution...")
        plt.figure(figsize=(10, 6))
        order = df['NAME_FAMILY_STATUS'].value_counts().index
        sns.countplot(data=df, x='NAME_FAMILY_STATUS', order=order, palette="Set3")
        plt.xticks(rotation=45)
        plt.title("Distribution of Family Status")
        plt.xlabel("Family Status")
        plt.ylabel("Count")
        plt.savefig(self.image_dir / "family_status_distribution.png", dpi=150)
        plt.close()

    def plot_credit_status_pie(self, labels_df: pd.DataFrame) -> None:
        """Plots the target variable (Credit Approval Status) distribution."""
        logger.info("Plotting Credit Status Distribution...")
        plt.figure(figsize=(8, 8))
        counts = labels_df['STATUS_LABEL'].value_counts()
        labels = ['Approved (Good)', 'Rejected (Bad)']
        colors = ['#2ca02c', '#d62728']  # Green for Good, Red for Bad
        
        plt.pie(counts, labels=labels, autopct='%1.2f%%', startangle=90, colors=colors, explode=(0, 0.1))
        plt.title("Credit Card Application Status Distribution (Target)")
        plt.savefig(self.image_dir / "credit_status_distribution.png", dpi=150)
        plt.close()

    def plot_demographic_boxplots(self, df: pd.DataFrame) -> None:
        """Plots Boxplots for income and age versus credit status."""
        logger.info("Plotting Demographic Boxplots...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Convert DAYS_BIRTH to positive age in years and map status to text labels
        df_copy = df.copy()
        df_copy['AGE'] = (-df_copy['DAYS_BIRTH']) / 365.25
        df_copy['STATUS_TEXT'] = df_copy['STATUS_LABEL'].map({0: 'Rejected', 1: 'Approved'})
        
        # 1. Income Boxplot vs Status
        # Limit y axis to 99th percentile for display
        upper_limit = df_copy['AMT_INCOME_TOTAL'].quantile(0.99)
        sns.boxplot(
            data=df_copy[df_copy['AMT_INCOME_TOTAL'] <= upper_limit], 
            x='STATUS_TEXT', 
            y='AMT_INCOME_TOTAL', 
            ax=axes[0], 
            palette={'Rejected': '#d62728', 'Approved': '#2ca02c'}
        )
        axes[0].set_title("Annual Income vs Application Status")
        axes[0].set_xlabel("Status")
        axes[0].set_ylabel("Income (USD)")
        
        # 2. Age Boxplot vs Status
        sns.boxplot(
            data=df_copy, 
            x='STATUS_TEXT', 
            y='AGE', 
            ax=axes[1], 
            palette={'Rejected': '#d62728', 'Approved': '#2ca02c'}
        )
        axes[1].set_title("Age vs Application Status")
        axes[1].set_xlabel("Status")
        axes[1].set_ylabel("Age (Years)")
        
        plt.savefig(self.image_dir / "demographic_boxplots.png", dpi=150)
        plt.close()

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Plots a correlation heatmap for numerical features."""
        logger.info("Plotting Correlation Heatmap...")
        plt.figure(figsize=(12, 10))
        
        # Filter only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Drop ID as it is an identifier, not a continuous variable
        if 'ID' in numerical_cols:
            numerical_cols.remove('ID')
            
        corr_matrix = df[numerical_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
        plt.title("Correlation Heatmap of Numerical Features")
        plt.savefig(self.image_dir / "correlation_heatmap.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    try:
        eda = ExploratoryDataAnalysis()
        eda.run_visualization_pipeline()
    except Exception as e:
        logger.error(f"EDA failed: {e}")
        sys.exit(1)
