-- =========================================================================
-- TABLE 1: Users
-- =========================================================================
CREATE TABLE IF NOT EXISTS Users (
    UserID      INTEGER PRIMARY KEY AUTOINCREMENT,
    Name        VARCHAR(100)  NOT NULL,
    Email       VARCHAR(150)  NOT NULL UNIQUE,
    Password    VARCHAR(255)  NOT NULL,
    Role        VARCHAR(50)   NOT NULL DEFAULT 'analyst'
);

SELECT * FROM Users;
-- =========================================================================
-- TABLE 2: Applicant_Details
-- (Expanded to store ALL form fields submitted during credit screening)
-- =========================================================================
CREATE TABLE IF NOT EXISTS Applicant_Details (
    ApplicantID        INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID             INTEGER       NOT NULL,
    Gender             VARCHAR(10)   NOT NULL DEFAULT 'Unknown',
    OwnCar             VARCHAR(5)    NOT NULL DEFAULT 'N',
    OwnRealty          VARCHAR(5)    NOT NULL DEFAULT 'N',
    ChildrenCount      INTEGER       NOT NULL DEFAULT 0,
    IncomeTotal        FLOAT         NOT NULL DEFAULT 0.0,
    IncomeType         VARCHAR(50)   NOT NULL,
    EducationType      VARCHAR(100)  NOT NULL,
    FamilyStatus       VARCHAR(50)   NOT NULL,
    HousingType        VARCHAR(50)   NOT NULL,
    DaysBirth          INTEGER       NOT NULL DEFAULT 0,
    EmploymentDays     INTEGER       NOT NULL,
    WorkPhone          INTEGER       NOT NULL DEFAULT 0,
    Phone              INTEGER       NOT NULL DEFAULT 0,
    EmailFlag          INTEGER       NOT NULL DEFAULT 0,
    OccupationType     VARCHAR(100)  NOT NULL DEFAULT 'Unknown',
    FamilyMembersCount FLOAT         NOT NULL DEFAULT 1.0,
    FOREIGN KEY (UserID) REFERENCES Users(UserID) 
        ON DELETE CASCADE 
        ON UPDATE CASCADE
);

SELECT * FROM Applicant_Details;

-- =========================================================================
-- TABLE 3: Credit_History
-- =========================================================================
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
SELECT * FROM Credit_History;

-- =========================================================================
-- TABLE 4: ML_Model
-- =========================================================================
CREATE TABLE IF NOT EXISTS ML_Model (
    ModelID         INTEGER PRIMARY KEY AUTOINCREMENT,
    ModelName       VARCHAR(100)  NOT NULL,
    AlgorithmType   VARCHAR(100)  NOT NULL,
    Accuracy        FLOAT         NOT NULL,
    MoreFile        VARCHAR(255)  DEFAULT NULL
);

SELECT * FROM ML_Model;
-- =========================================================================
-- TABLE 5: Approval_Prediction
-- =========================================================================
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

SELECT * FROM Approval_Prediction;
-- =========================================================================
-- SEED DATA & EXAMPLE INSERTS
-- =========================================================================

-- 1. Insert default user accounts (Parent Table)
INSERT INTO Users (Name, Email, Password, Role)
VALUES 
('Admin', 'admin@loanpredict.com', 'pbkdf2:sha256:admin2026', 'admin'),
('Analyst', 'analyst@loanpredict.com', 'pbkdf2:sha256:analyst2026', 'analyst');

-- 2. Insert ML Model meta logs (Parent Table)
INSERT INTO ML_Model (ModelName, AlgorithmType, Accuracy, MoreFile)
VALUES 
('XGBoost Classifier', 'XGBoost', 0.8829, 'best_model.pkl'),
('Ensemble Classifier (RF + XGB)', 'Ensemble', 0.8039, 'ensemble_model.pkl');

-- 3. Insert Applicant_Details (with all new columns populated)
-- (Requires a valid UserID. We use 2 to link this to the 'Analyst' user we created above)
INSERT INTO Applicant_Details (UserID, Gender, OwnCar, OwnRealty, ChildrenCount, IncomeTotal, IncomeType, EducationType, FamilyStatus, HousingType, DaysBirth, EmploymentDays, WorkPhone, Phone, EmailFlag, OccupationType, FamilyMembersCount)
VALUES 
(2, 'M', 'Y', 'Y', 0, 50000.00, 'Working', 'Higher education', 'Married', 'House / apartment', -10000, 1500, 1, 1, 1, 'Core staff', 2.0),
(2, 'F', 'N', 'N', 1, 35000.00, 'Commercial associate', 'Secondary / secondary special', 'Single / not married', 'Rented apartment', -8500, 800, 0, 1, 0, 'Sales staff', 2.0);

-- 4. Insert Credit_History 
-- (Requires a valid ApplicantID. We use 1 to link to the first applicant we just created)
INSERT INTO Credit_History (ApplicantID, MonthsBalance, PaymentStatus, OverdueStatus)
VALUES 
(1, 0, 'C', '0'),
(1, -1, 'C', '0'),
(1, -2, '1', '1'),
(2, 0, 'X', '0');

-- 5. Insert Approval_Prediction 
-- (Requires a valid ApplicantID and ModelID. We link Applicant 1 with Model 1 (XGBoost))
INSERT INTO Approval_Prediction (ApplicantID, ModelID, ApprovalResult, RiskCategory, PredictionDate)
VALUES 
(1, 1, 'Approved', 'Low Risk', CURRENT_TIMESTAMP),
(2, 1, 'Rejected', 'High Risk', CURRENT_TIMESTAMP);

