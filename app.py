import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FlaskBackend")

# Load configuration settings
from config import get_config
from src.prediction import CreditPredictor

config = get_config()

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config)

# Initialize SQLAlchemy Database
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# ==============================================================================
# DATABASE MODELS (MAPPED FROM ER DIAGRAM)
# ==============================================================================
# Entities: Users, Applicant_Details, Credit_History, ML_Model, Approval_Prediction
# Relationships:
#   Users (1) ──> (N) Applicant_Details
#   Applicant_Details (1) ──> (N) Credit_History
#   Applicant_Details (1) ──> (1) Approval_Prediction
#   ML_Model (1) ──> (N) Approval_Prediction
# ==============================================================================

class User(UserMixin, db.Model):
    """Users entity – represents system users (admin/analyst)."""
    __tablename__ = 'Users'

    UserID   = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Name     = db.Column(db.String(100), nullable=False)
    Email    = db.Column(db.String(150), nullable=False, unique=True)
    Password = db.Column(db.String(255), nullable=False)
    Role     = db.Column(db.String(50), nullable=False, default='analyst')

    # Relationship: Users (1) ──> (N) Applicant_Details
    applicants = db.relationship('ApplicantDetail', backref='user', lazy=True)

    def get_id(self):
        """Override for Flask-Login – returns the UserID as a string."""
        return str(self.UserID)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<User UserID={self.UserID} Name={self.Name}>"


@login_manager.user_loader
def load_user(user_id):
    """Flask-Login callback to load a user from the database by ID."""
    return db.session.get(User, int(user_id))


class ApplicantDetail(db.Model):
    """Applicant_Details entity – stores applicant personal & financial profile."""
    __tablename__ = 'Applicant_Details'

    ApplicantID      = db.Column(db.Integer, primary_key=True, autoincrement=True)
    UserID           = db.Column(db.Integer, db.ForeignKey('Users.UserID'), nullable=False)
    Gender           = db.Column(db.String(10), nullable=False, default='Unknown')
    OwnCar           = db.Column(db.String(5), nullable=False, default='N')
    OwnRealty        = db.Column(db.String(5), nullable=False, default='N')
    ChildrenCount    = db.Column(db.Integer, nullable=False, default=0)
    IncomeTotal      = db.Column(db.Float, nullable=False, default=0.0)
    IncomeType       = db.Column(db.String(50), nullable=False)
    EducationType    = db.Column(db.String(100), nullable=False)
    FamilyStatus     = db.Column(db.String(50), nullable=False)
    HousingType      = db.Column(db.String(50), nullable=False)
    DaysBirth        = db.Column(db.Integer, nullable=False, default=0)
    EmploymentDays   = db.Column(db.Integer, nullable=False)
    WorkPhone        = db.Column(db.Integer, nullable=False, default=0)
    Phone            = db.Column(db.Integer, nullable=False, default=0)
    EmailFlag        = db.Column(db.Integer, nullable=False, default=0)
    OccupationType   = db.Column(db.String(100), nullable=False, default='Unknown')
    FamilyMembersCount = db.Column(db.Float, nullable=False, default=1.0)

    # Relationship: Applicant_Details (1) ──> (N) Credit_History
    credit_histories = db.relationship('CreditHistory', backref='applicant', lazy=True)
    # Relationship: Applicant_Details (1) ──> (1) Approval_Prediction
    prediction = db.relationship('ApprovalPrediction', backref='applicant', uselist=False, lazy=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<ApplicantDetail ApplicantID={self.ApplicantID}>"


class CreditHistory(db.Model):
    """Credit_History entity – tracks credit repayment behavior per applicant."""
    __tablename__ = 'Credit_History'

    HistoryID     = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ApplicantID   = db.Column(db.Integer, db.ForeignKey('Applicant_Details.ApplicantID'), nullable=False)
    MonthsBalance = db.Column(db.Integer, nullable=False)
    PaymentStatus = db.Column(db.String(10), nullable=False)
    OverdueStatus = db.Column(db.String(10), nullable=False, default='0')

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<CreditHistory HistoryID={self.HistoryID} ApplicantID={self.ApplicantID}>"


class MLModel(db.Model):
    """ML_Model entity – stores machine learning model metadata."""
    __tablename__ = 'ML_Model'

    ModelID       = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ModelName     = db.Column(db.String(100), nullable=False)
    AlgorithmType = db.Column(db.String(100), nullable=False)
    Accuracy      = db.Column(db.Float, nullable=False)
    MoreFile      = db.Column(db.String(255), default=None)

    # Relationship: ML_Model (1) ──> (N) Approval_Prediction
    predictions = db.relationship('ApprovalPrediction', backref='model', lazy=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<MLModel ModelID={self.ModelID} Name={self.ModelName}>"


class ApprovalPrediction(db.Model):
    """Approval_Prediction entity – stores prediction outcomes linked to applicant & model."""
    __tablename__ = 'Approval_Prediction'

    PredictionID   = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ApplicantID    = db.Column(db.Integer, db.ForeignKey('Applicant_Details.ApplicantID'), nullable=False, unique=True)
    ModelID        = db.Column(db.Integer, db.ForeignKey('ML_Model.ModelID'), nullable=False)
    ApprovalResult = db.Column(db.String(20), nullable=False)
    RiskCategory   = db.Column(db.String(50), nullable=False)
    PredictionDate = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<ApprovalPrediction PredictionID={self.PredictionID} Result={self.ApprovalResult}>"

# ==============================================================================
# PIPELINE SETUP & MODEL CACHING
# ==============================================================================

# Global Predictor instance
predictor = None

def get_predictor() -> CreditPredictor:
    """Helper to lazy-load the prediction facade to accelerate app startup."""
    global predictor
    if predictor is None:
        predictor = CreditPredictor()
    return predictor


def initialize_database() -> None:
    """Creates tables and seeds default database data if empty."""
    with app.app_context():
        db.create_all()
        logger.info("Database tables verified/created successfully.")
        
        # Seed default users if none exist
        if User.query.first() is None:
            logger.info("Seeding default user accounts...")
            admin = User(
                Name="Admin",
                Email="admin@loanpredict.com",
                Password=generate_password_hash("admin2026"),
                Role="admin"
            )
            analyst = User(
                Name="Analyst",
                Email="analyst@loanpredict.com",
                Password=generate_password_hash("analyst2026"),
                Role="analyst"
            )
            db.session.add_all([admin, analyst])
            db.session.commit()
            
        # Seed Champion Model metadata if none exists
        if MLModel.query.first() is None:
            logger.info("Seeding ML Model meta logs...")
            model_record = MLModel(
                ModelName="XGBoost Classifier",
                AlgorithmType="XGBoost",
                Accuracy=0.8829,
                MoreFile="best_model.pkl"
            )
            db.session.add(model_record)
            db.session.commit()

        # Add new Ensemble Classifier metadata if it doesn't exist
        ensemble_model = MLModel.query.filter_by(ModelName="Ensemble Classifier (RF + XGB)").first()
        if ensemble_model is None:
            logger.info("Adding new Ensemble Classifier metadata to DB...")
            ensemble_record = MLModel(
                ModelName="Ensemble Classifier (RF + XGB)",
                AlgorithmType="Ensemble",
                Accuracy=0.8039,
                MoreFile="ensemble_model.pkl"
            )
            db.session.add(ensemble_record)
            db.session.commit()

# ==============================================================================
# AUTHENTICATION ROUTES
# ==============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Serves the login page and handles user authentication."""
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please fill in all fields.', 'warning')
            return redirect(url_for('login'))

        user = User.query.filter_by(Email=email).first()

        if user and check_password_hash(user.Password, password):
            login_user(user)
            logger.info(f"User '{user.Name}' logged in successfully.")
            flash(f'Welcome back, {user.Name}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Serves the registration page and creates new user accounts."""
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        if not name or not email or not password:
            flash('All fields are required.', 'warning')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'warning')
            return redirect(url_for('register'))

        # Check if user already exists
        existing_user = User.query.filter_by(Email=email).first()
        if existing_user:
            flash('An account with this email already exists.', 'danger')
            return redirect(url_for('register'))

        # Create new user
        new_user = User(
            Name=name,
            Email=email,
            Password=generate_password_hash(password),
            Role='analyst'
        )
        db.session.add(new_user)
        db.session.commit()

        logger.info(f"New user '{name}' registered with email '{email}'.")
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    """Logs the user out and redirects to login page."""
    logger.info(f"User '{current_user.Name}' logged out.")
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ==============================================================================
# WEB ROUTING
# ==============================================================================

@app.route('/')
@login_required
def home():
    """Renders the dashboard landing page."""
    try:
        # Load stats
        total_applicants = ApplicantDetail.query.count()
        
        # Calculate approval metrics
        total_predictions = ApprovalPrediction.query.count()
        approved_count = ApprovalPrediction.query.filter_by(ApprovalResult='Approved').count()
        rejected_count = ApprovalPrediction.query.filter_by(ApprovalResult='Rejected').count()
        
        approval_rate = (approved_count / total_predictions * 100) if total_predictions > 0 else 0.0
        rejection_rate = (rejected_count / total_predictions * 100) if total_predictions > 0 else 0.0
        
        # Get active model info (use the first model by default)
        active_model = MLModel.query.first()
        model_name = active_model.ModelName if active_model else "N/A"
        model_accuracy = active_model.Accuracy if active_model else 0.0
        
        # Recent prediction logs with full applicant and prediction data
        recent_predictions = (
            db.session.query(ApprovalPrediction, ApplicantDetail)
            .join(ApplicantDetail, ApprovalPrediction.ApplicantID == ApplicantDetail.ApplicantID)
            .order_by(ApprovalPrediction.PredictionDate.desc())
            .limit(5)
            .all()
        )
        
        stats = {
            "total_applicants": total_applicants,
            "total_predictions": total_predictions,
            "approval_rate": round(approval_rate, 2),
            "rejection_rate": round(rejection_rate, 2),
            "model_name": model_name,
            "model_accuracy": round(model_accuracy * 100, 2),
            "model_f1": round(model_accuracy * 100, 2),
            "recent_predictions": recent_predictions
        }
        
        return render_template('home.html', stats=stats)
    except Exception as e:
        logger.error(f"Error serving dashboard homepage: {e}")
        return f"Database is initializing or encountered an error. Details: {e}", 500


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Serves the applicant prediction form and processes predictions."""
    if request.method == 'GET':
        return render_template('index.html')
        
    try:
        # 1. Fetch form inputs
        form = request.form
            
        # Parse age from birthday
        birthday_str = form.get('birthday')
        birthday_date = datetime.strptime(birthday_str, "%Y-%m-%d")
        delta = datetime.today() - birthday_date
        age_years = delta.days / 365.25
        days_birth = -delta.days
        
        # Parse employment details
        is_unemployed_input = int(form.get('is_unemployed', 0))
        if is_unemployed_input == 1:
            days_employed = 365243
            years_employed = 0.0
        else:
            employed_date_str = form.get('employment_start')
            employed_date = datetime.strptime(employed_date_str, "%Y-%m-%d")
            employed_delta = datetime.today() - employed_date
            days_employed = -employed_delta.days
            years_employed = employed_delta.days / 365.25
            
        # Compile raw profile dictionary for the predictor
        applicant_data_dict = {
            'CODE_GENDER': form.get('gender'),
            'FLAG_OWN_CAR': form.get('own_car'),
            'FLAG_OWN_REALTY': form.get('own_realty'),
            'CNT_CHILDREN': int(form.get('children_count', 0)),
            'AMT_INCOME_TOTAL': float(form.get('income_total', 0)),
            'NAME_INCOME_TYPE': form.get('income_type'),
            'NAME_EDUCATION_TYPE': form.get('education_type'),
            'NAME_FAMILY_STATUS': form.get('family_status'),
            'NAME_HOUSING_TYPE': form.get('housing_type'),
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'FLAG_WORK_PHONE': int(form.get('work_phone', 0)),
            'FLAG_PHONE': int(form.get('phone', 0)),
            'FLAG_EMAIL': int(form.get('email', 0)),
            'OCCUPATION_TYPE': form.get('occupation_type') or 'Unknown',
            'CNT_FAM_MEMBERS': float(form.get('family_members_count', 1))
        }
        
        # 2. Write applicant profile records to DB with ALL form fields
        applicant = ApplicantDetail(
            UserID=current_user.UserID,
            Gender=applicant_data_dict['CODE_GENDER'],
            OwnCar=applicant_data_dict['FLAG_OWN_CAR'],
            OwnRealty=applicant_data_dict['FLAG_OWN_REALTY'],
            ChildrenCount=applicant_data_dict['CNT_CHILDREN'],
            IncomeTotal=applicant_data_dict['AMT_INCOME_TOTAL'],
            IncomeType=applicant_data_dict['NAME_INCOME_TYPE'],
            EducationType=applicant_data_dict['NAME_EDUCATION_TYPE'],
            FamilyStatus=applicant_data_dict['NAME_FAMILY_STATUS'],
            HousingType=applicant_data_dict['NAME_HOUSING_TYPE'],
            DaysBirth=days_birth,
            EmploymentDays=days_employed,
            WorkPhone=applicant_data_dict['FLAG_WORK_PHONE'],
            Phone=applicant_data_dict['FLAG_PHONE'],
            EmailFlag=applicant_data_dict['FLAG_EMAIL'],
            OccupationType=applicant_data_dict['OCCUPATION_TYPE'],
            FamilyMembersCount=applicant_data_dict['CNT_FAM_MEMBERS']
        )
        db.session.add(applicant)
        db.session.commit()
        
        # Seed a credit history record for the applicant
        credit_history = CreditHistory(
            ApplicantID=applicant.ApplicantID,
            MonthsBalance=0,
            PaymentStatus='X',
            OverdueStatus='0'
        )
        db.session.add(credit_history)
        db.session.commit()
        
        # 3. Trigger Machine Learning Prediction Facade
        predictor_service = get_predictor()
        prediction, confidence = predictor_service.predict_approval(applicant_data_dict)
        
        # 4. Save prediction outcomes to Approval_Prediction table
        active_model = MLModel.query.first()
        model_id = active_model.ModelID if active_model else 1
        
        # Determine risk category based on confidence
        if confidence >= 0.8:
            risk_category = "Low Risk"
        elif confidence >= 0.5:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
        
        prediction_text = "Approved" if prediction == 1 else "Rejected"
        
        prediction_record = ApprovalPrediction(
            ApplicantID=applicant.ApplicantID,
            ModelID=model_id,
            ApprovalResult=prediction_text,
            RiskCategory=risk_category
        )
        db.session.add(prediction_record)
        db.session.commit()
        
        logger.info(f"Prediction saved: Applicant #{applicant.ApplicantID} -> {prediction_text} (Confidence: {confidence:.4f})")
        
        # Render the result template with details
        confidence_percent = round(confidence * 100, 2)
        
        return render_template(
            'result.html', 
            prediction=prediction_text.upper(), 
            confidence=confidence_percent,
            applicant_id=applicant.ApplicantID,
            income=f"{applicant_data_dict['AMT_INCOME_TOTAL']:,.2f}"
        )
        
    except Exception as e:
        logger.error(f"Error handling prediction submission: {e}")
        flash(f"Error processing submission. Details: {e}", "danger")
        return redirect(url_for('predict'))


if __name__ == "__main__":
    # Ensure database tables and seeds are initialized before launch
    initialize_database()
    
    # Run the waitress production server on Windows, or standard dev server
    env = os.getenv("FLASK_ENV", "development").lower()
    if env == "production":
        logger.info("Starting Waitress production WSGI server on port 5000...")
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)
    else:
        logger.info("Starting Flask development server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=True)

