"""
Appointment System for Smart Health Report Analyzer
Handles scheduling, rescheduling, and cancellation of patient appointments
"""

from datetime import date, time
from typing import Optional, List, Dict
import logging
from database_system import get_database

logger = logging.getLogger(__name__)


class AppointmentSystem:
    """Manages patient appointments by delegating database operations."""

    def __init__(self):
        """Initialize appointment system with a database connection."""
        self.db = get_database()

    def create_appointment(self, patient_id: int, doctor_name: str,
                             appointment_date: date, appointment_time: time,
                             specialization: str = None, purpose: str = None,
                             hospital: str = None) -> Optional[int]:
        """Create a new appointment after checking for conflicts."""
        if self._check_conflict(patient_id, appointment_date, appointment_time):
            logger.warning(f"Appointment conflict for patient {patient_id} at {appointment_date} {appointment_time}")
            return None

        query = """
        INSERT INTO appointment_data 
        (patient_id, doctor_name, specialization, appointment_date, 
         appointment_time, purpose, hospital, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'scheduled')
        """
        params = (patient_id, doctor_name, specialization, appointment_date, 
                  appointment_time, purpose, hospital)
        
        appointment_id = self.db.execute_commit(query, params)
        if appointment_id:
            logger.info(f"Appointment created: ID {appointment_id}")
        return appointment_id

    def _check_conflict(self, patient_id: int, appointment_date: date, appointment_time: time) -> bool:
        """Check if the patient already has a scheduled appointment at the given time."""
        query = """
        SELECT COUNT(*) as count FROM appointment_data
        WHERE patient_id = %s AND appointment_date = %s AND appointment_time = %s AND status = 'scheduled'
        """
        result = self.db.fetch_one(query, (patient_id, appointment_date, appointment_time))
        return result['count'] > 0 if result else False

    def get_patient_appointments(self, patient_id: int) -> List[Dict]:
        """Get all appointments for a specific patient."""
        query = "SELECT * FROM appointment_data WHERE patient_id = %s ORDER BY appointment_date DESC, appointment_time DESC"
        return self.db.fetch_all(query, (patient_id,))

    def cancel_appointment(self, appointment_id: int) -> bool:
        """Cancel an appointment by updating its status."""
        query = "UPDATE appointment_data SET status = 'cancelled' WHERE appointment_id = %s"
        result_id = self.db.execute_commit(query, (appointment_id,))
        # In this case, execute_commit doesn't return a useful ID, but we check if it ran without error (is not None)
        # A more robust check might be to see if rows were affected.
        if result_id is not None:
             logger.info(f"Appointment {appointment_id} cancelled")
             return True
        return False