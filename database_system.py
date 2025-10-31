"""
Database System for Smart Health Report Analyzer
Manages MySQL connections and all database operations.
This version includes automatic reconnection and is hardened against connection failures.
"""

import mysql.connector  # type: ignore
from mysql.connector import Error # type: ignore
import logging
import json
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSystem:
    """Manages database connections and operations with auto-reconnect and safety checks."""

    def __init__(self, host: str = "localhost", user: str = "root",
                 password: str = "Gokul127!", database: str = "health_analyzer"):
        self.host = host
        self.user = user
        self.password = password
        self.database_name = database
        self.connection = None
        self.connect()

    def connect(self):
        """Establishes a new connection to the MySQL database."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL server.")
                self._create_database()
                self._create_tables()
        except Error as e:
            logger.error(f"FATAL: Could not connect to MySQL. Check server status and credentials. Error: {e}")
            self.connection = None

    def reconnect(self):
        """Checks if the connection is active, and reconnects if it's lost or was never established."""
        try:
            if self.connection is None or not self.connection.is_connected():
                logger.warning("Database connection is not available. Attempting to reconnect...")
                self.connect()
        except Error as e:
            logger.error(f"Reconnect failed: {e}")
            self.connection = None

    def _create_database(self):
        """Creates the database if it doesn't exist."""
        if not self.connection: return
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database_name}")
                cursor.execute(f"USE {self.database_name}")
        except Error as e:
            logger.error(f"Error creating/selecting database: {e}")

    def _create_tables(self):
        """Creates all necessary tables if they don't exist."""
        if not self.connection: return

        # FIX: Added weight and height columns to the patient_data table definition.
        patient_table = """
        CREATE TABLE IF NOT EXISTS patient_data (
            patient_id INT AUTO_INCREMENT PRIMARY KEY,
            login_id VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            patient_name VARCHAR(200) NOT NULL,
            age INT,
            weight FLOAT,
            height FLOAT,
            gender VARCHAR(10),
            phone VARCHAR(20),
            email VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
        
        appointment_table = "CREATE TABLE IF NOT EXISTS appointment_data (appointment_id INT AUTO_INCREMENT PRIMARY KEY, patient_id INT NOT NULL, doctor_name VARCHAR(200) NOT NULL, specialization VARCHAR(100), appointment_date DATE NOT NULL, appointment_time TIME NOT NULL, purpose TEXT, status VARCHAR(50) DEFAULT 'scheduled', hospital VARCHAR(200), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (patient_id) REFERENCES patient_data(patient_id) ON DELETE CASCADE)"
        reports_table = "CREATE TABLE IF NOT EXISTS medical_reports (report_id INT AUTO_INCREMENT PRIMARY KEY, patient_id INT NOT NULL, report_type VARCHAR(100), report_text LONGTEXT, predictions JSON, risk_level VARCHAR(50), uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (patient_id) REFERENCES patient_data(patient_id) ON DELETE CASCADE)"
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(patient_table)
                cursor.execute(appointment_table)
                cursor.execute(reports_table)
            self.connection.commit()
        except Error as e:
            logger.error(f"Error creating tables: {e}")

    # --- Robust Helper Methods ---

    def execute_commit(self, query: str, params: tuple = None) -> Optional[int]:
        self.reconnect()
        if not self.connection:
            logger.error("Cannot execute query: no database connection.")
            return None
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or ())
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            logger.error(f"Query failed: {e}")
            return None

    def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        self.reconnect()
        if not self.connection: return []
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute(query, params or ())
                return cursor.fetchall()
        except Error as e:
            logger.error(f"Query failed: {e}")
            return []

    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        self.reconnect()
        if not self.connection: return None
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute(query, params or ())
                return cursor.fetchone()
        except Error as e:
            logger.error(f"Query failed: {e}")
            return None

    # --- Public Data Methods ---

    def register_patient(self, **kwargs) -> Optional[int]:
        """
        Dynamically builds and executes an INSERT statement for a new patient.
        This handles all optional fields gracefully.
        """
        columns = ['login_id', 'password_hash', 'patient_name']
        values = [kwargs.get('login_id'), kwargs.get('password_hash'), kwargs.get('patient_name')]
        
        # FIX: Added weight and height to the list of optional fields.
        optional_fields = ['age', 'weight', 'height', 'gender', 'phone', 'email']
        for field in optional_fields:
            if field in kwargs and kwargs[field] is not None:
                columns.append(field)
                values.append(kwargs[field])
        
        columns_str = ', '.join(columns)
        placeholders_str = ', '.join(['%s'] * len(values))
        query = f"INSERT INTO patient_data ({columns_str}) VALUES ({placeholders_str})"
        
        return self.execute_commit(query, tuple(values))

    def authenticate_patient(self, login_id: str, password_hash: str) -> Optional[Dict]:
        query = "SELECT * FROM patient_data WHERE login_id = %s AND password_hash = %s"
        return self.fetch_one(query, (login_id, password_hash))

    def get_patient_reports(self, patient_id: int) -> List[Dict]:
        query = "SELECT * FROM medical_reports WHERE patient_id = %s ORDER BY uploaded_at DESC"
        return self.fetch_all(query, (patient_id,))

    def save_medical_report(self, **kwargs) -> Optional[int]:
        query = "INSERT INTO medical_reports (patient_id, report_type, report_text, predictions, risk_level) VALUES (%s, %s, %s, %s, %s)"
        params = (kwargs.get('patient_id'), kwargs.get('report_type'), kwargs.get('report_text'), json.dumps(kwargs.get('predictions')), kwargs.get('risk_level'))
        return self.execute_commit(query, params)

    def close(self):
        """Closes the database connection if it exists."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed.")


# Singleton instance
_db_instance = None

def get_database() -> DatabaseSystem:
    """Gets the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseSystem()
    return _db_instance