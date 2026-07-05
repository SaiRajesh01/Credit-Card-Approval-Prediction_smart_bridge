import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash

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

# ==============================================================================
# DATABASE MODELS (MAPPED FROM ER DIAGRAM)
# ==============================================================================

class User(db.Model):
    """User entity representing the bank credit analyst."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    applicants = db.relationship('ApplicantDetail', backref='analyst', lazy=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<User {self.username}>"


class ApplicantDetail(db.Model):
    """Applicant_Details entity representing applicant personal and financial profile."""
    __tablename__ = 'applicant_details'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Demographic fields matching application_record.csv
    gender = db.Column(db.String(10), nullable=False)
    own_car = db.Column(db.String(10), nullable=False)
    own_realty = db.Column(db.String(10), nullable=False)
    children_count = db.Column(db.Integer, nullable=False)
    income_total = db.Column(db.Float, nullable=False)
    income_type = db.Column(db.String(50), nullable=False)
    education_type = db.Column(db.String(100), nullable=False)
    family_status = db.Column(db.String(50), nullable=False)
    housing_type = db.Column(db.String(50), nullable=False)
    
    # Engineered fields
    age_years = db.Column(db.Float, nullable=False)
    years_employed = db.Column(db.Float, nullable=False)
    is_unemployed = db.Column(db.Integer, nullable=False)
    
    # Communication flags
    work_phone = db.Column(db.Integer, nullable=False)
    phone = db.Column(db.Integer, nullable=False)
    email = db.Column(db.Integer, nullable=False)
    
    # Occupation and Family details
    occupation_type = db.Column(db.String(50), nullable=False)
    family_members_count = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    credit_histories = db.relationship('CreditHistory', backref='applicant', lazy=True)
    predictions = db.relationship('ApprovalPrediction', backref='applicant', lazy=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<ApplicantDetail ID={self.id}>"


class CreditHistory(db.Model):
    """Credit_History entity representing credit repayment behavior."""
    __tablename__ = 'credit_history'
    
    id = db.Column(db.Integer, primary_key=True)
    applicant_id = db.Column(db.Integer, db.ForeignKey('applicant_details.id'), nullable=False)
    months_balance = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(5), nullable=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<CreditHistory ApplicantID={self.applicant_id} Status={self.status}>"


class MLModel(db.Model):
    """ML_Model entity representing the serialized estimator metrics."""
    __tablename__ = 'ml_models'
    
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    version = db.Column(db.String(20), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    active = db.Column(db.Boolean, default=True)
    
    # Relationships
    predictions = db.relationship('ApprovalPrediction', backref='model', lazy=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<MLModel {self.model_name} v{self.version}>"


class ApprovalPrediction(db.Model):
    """Approval_Prediction entity containing model prediction outcomes and audits."""
    __tablename__ = 'approval_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    applicant_id = db.Column(db.Integer, db.ForeignKey('applicant_details.id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'), nullable=False)
    prediction = db.Column(db.Integer, nullable=False)  # 1 = Approved, 0 = Rejected
    confidence_score = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<ApprovalPrediction ID={self.id} Outcome={self.prediction}>"

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
        
        # Seed default Credit Analyst User if none exists
        if User.query.first() is None:
            logger.info("Seeding default analyst user account...")
            hashed_pwd = generate_password_hash("analyst2026")
            analyst = User(username="analyst", password_hash=hashed_pwd)
            db.session.add(analyst)
            db.session.commit()
            
        # Seed Champion Model metadata if none exists
        if MLModel.query.first() is None:
            logger.info("Seeding ML Model meta logs...")
            model_record = MLModel(
                model_name="XGBoost Classifier",
                version="1.0.0",
                accuracy=0.8829,
                f1_score=0.9377,
                active=True
            )
            db.session.add(model_record)
            db.session.commit()

        # Add new Ensemble Classifier metadata if it doesn't exist
        ensemble_model = MLModel.query.filter_by(model_name="Ensemble Classifier (RF + XGB)").first()
        if ensemble_model is None:
            logger.info("Adding new Ensemble Classifier metadata to DB...")
            # Set all other models to inactive
            MLModel.query.update({MLModel.active: False})
            
            ensemble_record = MLModel(
                model_name="Ensemble Classifier (RF + XGB)",
                version="1.1.0",
                accuracy=0.8039,
                f1_score=0.8801,
                active=True
            )
            db.session.add(ensemble_record)
            db.session.commit()

# ==============================================================================
# WEB ROUTING
# ==============================================================================

@app.route('/')
def home():
    """Renders the dashboard landing page."""
    try:
        # Load stats
        total_applicants = ApplicantDetail.query.count()
        
        # Calculate approval metrics
        total_predictions = ApprovalPrediction.query.count()
        approved_count = ApprovalPrediction.query.filter_by(prediction=1).count()
        rejected_count = ApprovalPrediction.query.filter_by(prediction=0).count()
        
        approval_rate = (approved_count / total_predictions * 100) if total_predictions > 0 else 0.0
        rejection_rate = (rejected_count / total_predictions * 100) if total_predictions > 0 else 0.0
        
        # Get active model info
        active_model = MLModel.query.filter_by(active=True).first()
        model_name = active_model.model_name if active_model else "N/A"
        model_f1 = active_model.f1_score if active_model else 0.0
        
        # Recent prediction logs
        recent_predictions = (
            db.session.query(ApprovalPrediction, ApplicantDetail)
            .join(ApplicantDetail, ApprovalPrediction.applicant_id == ApplicantDetail.id)
            .order_by(ApprovalPrediction.created_at.desc())
            .limit(5)
            .all()
        )
        
        stats = {
            "total_applicants": total_applicants,
            "total_predictions": total_predictions,
            "approval_rate": round(approval_rate, 2),
            "rejection_rate": round(rejection_rate, 2),
            "model_name": model_name,
            "model_f1": round(model_f1 * 100, 2),
            "recent_predictions": recent_predictions
        }
        
        return render_template('home.html', stats=stats)
    except Exception as e:
        logger.error(f"Error serving dashboard homepage: {e}")
        return f"Database is initializing or encountered an error. Details: {e}", 500


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Serves the applicant prediction form and processes predictions."""
    if request.method == 'GET':
        return render_template('index.html')
        
    try:
        # 1. Fetch form inputs
        form = request.form
        
        # Get default analyst user to link applicant
        analyst = User.query.filter_by(username="analyst").first()
        if not analyst:
            flash("User authentication error.", "danger")
            return redirect(url_for('predict'))
            
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
        
        # 2. Write applicant profile records to DB
        applicant = ApplicantDetail(
            user_id=analyst.id,
            gender=applicant_data_dict['CODE_GENDER'],
            own_car=applicant_data_dict['FLAG_OWN_CAR'],
            own_realty=applicant_data_dict['FLAG_OWN_REALTY'],
            children_count=applicant_data_dict['CNT_CHILDREN'],
            income_total=applicant_data_dict['AMT_INCOME_TOTAL'],
            income_type=applicant_data_dict['NAME_INCOME_TYPE'],
            education_type=applicant_data_dict['NAME_EDUCATION_TYPE'],
            family_status=applicant_data_dict['NAME_FAMILY_STATUS'],
            housing_type=applicant_data_dict['NAME_HOUSING_TYPE'],
            age_years=round(age_years, 2),
            years_employed=round(years_employed, 2),
            is_unemployed=is_unemployed_input,
            work_phone=applicant_data_dict['FLAG_WORK_PHONE'],
            phone=applicant_data_dict['FLAG_PHONE'],
            email=applicant_data_dict['FLAG_EMAIL'],
            occupation_type=applicant_data_dict['OCCUPATION_TYPE'],
            family_members_count=applicant_data_dict['CNT_FAM_MEMBERS']
        )
        db.session.add(applicant)
        db.session.commit()
        
        # Seed a simple mock payment balance record in credit history (months_balance=0, status=X)
        credit_history = CreditHistory(
            applicant_id=applicant.id,
            months_balance=0,
            status='X'
        )
        db.session.add(credit_history)
        db.session.commit()
        
        # 3. Trigger Machine Learning Prediction Facade
        predictor_service = get_predictor()
        prediction, confidence = predictor_service.predict_approval(applicant_data_dict)
        
        # 4. Save prediction outcomes to prediction audit tables
        active_model = MLModel.query.filter_by(active=True).first()
        model_id = active_model.id if active_model else 1
        
        prediction_record = ApprovalPrediction(
            applicant_id=applicant.id,
            model_id=model_id,
            prediction=prediction,
            confidence_score=confidence
        )
        db.session.add(prediction_record)
        db.session.commit()
        
        # Render the result template with details
        prediction_text = "APPROVED" if prediction == 1 else "REJECTED"
        confidence_percent = round(confidence * 100, 2)
        
        return render_template(
            'result.html', 
            prediction=prediction_text, 
            confidence=confidence_percent,
            applicant_id=applicant.id,
            income=f"{applicant.income_total:,.2f}"
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
