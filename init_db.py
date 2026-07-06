"""
SQL Schema Initialization Script for credit_card.db
Based on the ER Diagram for the Loan Approval Prediction System.

Entities:
  1. Users
  2. Applicant_Details
  3. Credit_History
  4. ML_Model
  5. Approval_Prediction

Run this script to recreate the database with the proper schema.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_card.db")


def create_database():
    """Drop all existing tables and recreate them matching the ER diagram."""

    # Remove existing DB file to start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[INFO] Removed existing database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    # =========================================================================
    # TABLE 1: Users
    # Primary Key: UserID
    # Attributes: UserID, Name, Email, Password, Role
    # =========================================================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Users (
        UserID      INTEGER PRIMARY KEY AUTOINCREMENT,
        Name        VARCHAR(100)  NOT NULL,
        Email       VARCHAR(150)  NOT NULL UNIQUE,
        Password    VARCHAR(255)  NOT NULL,
        Role        VARCHAR(50)   NOT NULL DEFAULT 'analyst'
    );
    """)
    print("[OK] Created table: Users")

    # =========================================================================
    # TABLE 2: Applicant_Details
    # Primary Key: ApplicantID
    # Foreign Key: UserID -> Users(UserID)
    # Attributes: ApplicantID, UserID, IncomeType, EducationType,
    #             FamilyStatus, HousingType, EmploymentDays
    # Relationship: Users (1) --> (N) Applicant_Details
    # =========================================================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Applicant_Details (
        ApplicantID     INTEGER PRIMARY KEY AUTOINCREMENT,
        UserID          INTEGER       NOT NULL,
        IncomeType      VARCHAR(50)   NOT NULL,
        EducationType   VARCHAR(100)  NOT NULL,
        FamilyStatus    VARCHAR(50)   NOT NULL,
        HousingType     VARCHAR(50)   NOT NULL,
        EmploymentDays  INTEGER       NOT NULL,
        FOREIGN KEY (UserID) REFERENCES Users(UserID)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );
    """)
    print("[OK] Created table: Applicant_Details")

    # =========================================================================
    # TABLE 3: Credit_History
    # Primary Key: HistoryID
    # Foreign Key: ApplicantID -> Applicant_Details(ApplicantID)
    # Attributes: HistoryID, ApplicantID, MonthsBalance,
    #             PaymentStatus, OverdueStatus
    # Relationship: Applicant_Details (1) --> (N) Credit_History
    # =========================================================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Credit_History (
        HistoryID       INTEGER PRIMARY KEY AUTOINCREMENT,
        ApplicantID     INTEGER       NOT NULL,
        MonthsBalance   INTEGER       NOT NULL,
        PaymentStatus   VARCHAR(10)   NOT NULL,
        OverdueStatus   VARCHAR(10)   NOT NULL DEFAULT '0',
        FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details(ApplicantID)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );
    """)
    print("[OK] Created table: Credit_History")

    # =========================================================================
    # TABLE 4: ML_Model
    # Primary Key: ModelID
    # Attributes: ModelID, ModelName, AlgorithmType, Accuracy, MoreFile
    # =========================================================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ML_Model (
        ModelID         INTEGER PRIMARY KEY AUTOINCREMENT,
        ModelName       VARCHAR(100)  NOT NULL,
        AlgorithmType   VARCHAR(100)  NOT NULL,
        Accuracy        FLOAT         NOT NULL,
        MoreFile        VARCHAR(255)  DEFAULT NULL
    );
    """)
    print("[OK] Created table: ML_Model")

    # =========================================================================
    # TABLE 5: Approval_Prediction
    # Primary Key: PredictionID
    # Foreign Keys:
    #   ApplicantID -> Applicant_Details(ApplicantID)
    #   ModelID     -> ML_Model(ModelID)
    # Attributes: PredictionID, ApplicantID, ModelID,
    #             ApprovalResult, RiskCategory, PredictionDate
    # Relationships:
    #   Applicant_Details (1) --> (1) Approval_Prediction
    #   ML_Model (1) --> (N) Approval_Prediction
    # =========================================================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Approval_Prediction (
        PredictionID    INTEGER PRIMARY KEY AUTOINCREMENT,
        ApplicantID     INTEGER       NOT NULL UNIQUE,
        ModelID         INTEGER       NOT NULL,
        ApprovalResult  VARCHAR(20)   NOT NULL,
        RiskCategory    VARCHAR(50)   NOT NULL,
        PredictionDate  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details(ApplicantID)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
        FOREIGN KEY (ModelID) REFERENCES ML_Model(ModelID)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );
    """)
    print("[OK] Created table: Approval_Prediction")

    conn.commit()

    # =========================================================================
    # SEED DATA
    # =========================================================================
    print("\n[INFO] Seeding initial data...")

    # Seed a default admin user
    cursor.execute("""
    INSERT INTO Users (Name, Email, Password, Role)
    VALUES ('Admin', 'admin@loanpredict.com', 'pbkdf2:sha256:admin2026', 'admin');
    """)

    # Seed a default analyst user
    cursor.execute("""
    INSERT INTO Users (Name, Email, Password, Role)
    VALUES ('Analyst', 'analyst@loanpredict.com', 'pbkdf2:sha256:analyst2026', 'analyst');
    """)
    print("[OK] Seeded Users table")

    # Seed ML Model records
    cursor.execute("""
    INSERT INTO ML_Model (ModelName, AlgorithmType, Accuracy, MoreFile)
    VALUES ('XGBoost Classifier', 'XGBoost', 0.8829, 'best_model.pkl');
    """)

    cursor.execute("""
    INSERT INTO ML_Model (ModelName, AlgorithmType, Accuracy, MoreFile)
    VALUES ('Ensemble Classifier (RF + XGB)', 'Ensemble', 0.8039, 'ensemble_model.pkl');
    """)
    print("[OK] Seeded ML_Model table")

    conn.commit()

    # =========================================================================
    # VERIFICATION: Print all table schemas and row counts
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATABASE SCHEMA VERIFICATION")
    print("=" * 60)

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()

    for (table_name,) in tables:
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table_name}'")
        schema = cursor.fetchone()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"\n--- {table_name} ({count} rows) ---")
        print(schema[0])

    conn.close()
    print(f"\n[SUCCESS] Database created at: {DB_PATH}")


if __name__ == "__main__":
    create_database()
