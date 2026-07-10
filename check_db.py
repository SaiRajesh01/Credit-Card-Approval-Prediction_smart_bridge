import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_card.db")

def check_database():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    tables = [
        "Users",
        "Applicant_Details",
        "Credit_History",
        "ML_Model",
        "Approval_Prediction"
    ]

    print("=" * 50)
    print("DATABASE CONTENTS")
    print("=" * 50)

    for table in tables:
        print(f"\n--- Table: {table} ---")
        try:
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            
            if not rows:
                print("(Empty - No data yet)")
            else:
                # Print column names
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                print(" | ".join(columns))
                print("-" * 30)
                
                # Print rows
                for row in rows:
                    print(row)
                    
        except sqlite3.OperationalError as e:
            print(f"Error reading table {table}: {e}")

    conn.close()

if __name__ == "__main__":
    check_database()
