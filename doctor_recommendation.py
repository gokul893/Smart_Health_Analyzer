"""
Doctor Recommendation System
Recommends doctors based on predicted diseases and specializations
"""

import pandas as pd
import logging
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)

class DoctorRecommendationSystem:
    """Recommends doctors based on disease predictions and specializations"""
    
    def __init__(self, doctors_data_path: str = "doctors_database.csv"):
        """
        Initialize doctor recommendation system
        
        Args:
            doctors_data_path: Path to doctors database CSV
        """
        self.doctors_data_path = doctors_data_path
        self.doctors_df = None
        self._load_doctors_database()

        self.disease_specialization_map = {
            'heart_disease': 'Cardiologist',
            'diabetes': 'Endocrinologist',
            'liver': 'Hepatologist',
            'kidney_disease': 'Nephrologist',
            'thyroid': 'Endocrinologist'
        }
    
    def _load_doctors_database(self):
        """Load doctors database from CSV or create sample data"""
        try:
            if os.path.exists(self.doctors_data_path):
                self.doctors_df = pd.read_csv(self.doctors_data_path)
                logger.info(f"Loaded doctors database with {len(self.doctors_df)} doctors")
            else:
                # Create sample doctors database
                self._create_sample_database()
                logger.info("Created sample doctors database")
                
        except Exception as e:
            logger.error(f"Error loading doctors database: {e}")
            self._create_sample_database()
    
    def _create_sample_database(self):
        """Create sample doctors database for demonstration"""
        
        sample_doctors = [
            # Cardiologists
            {
                'doctor_id': 1,
                'name': 'Dr. Rajesh Kumar',
                'specialization': 'Cardiologist',
                'hospital': 'SRM Hospital',
                'experience_years': 15,
                'availability': 'Mon-Fri 9AM-5PM',
                'rating': 4.8,
                'consultation_fee': 1000,
                'phone': '+91-9876543210',
                'email': 'rajesh.kumar@apollo.com'
            },
            {
                'doctor_id': 2,
                'name': 'Dr. Priya Sharma',
                'specialization': 'Cardiologist',
                'hospital': 'SRM Hospital',
                'experience_years': 12,
                'availability': 'Mon-Sat 10AM-6PM',
                'rating': 4.7,
                'consultation_fee': 1200,
                'phone': '+91-9876543211',
                'email': 'priya.sharma@fortis.com'
            },
            
            # Endocrinologists
            {
                'doctor_id': 3,
                'name': 'Dr. Amit Patel',
                'specialization': 'Endocrinologist',
                'hospital': 'SRM Hospital',
                'experience_years': 18,
                'availability': 'Tue-Sat 9AM-4PM',
                'rating': 4.9,
                'consultation_fee': 1500,
                'phone': '+91-9876543212',
                'email': 'amit.patel@max.com'
            },
            {
                'doctor_id': 4,
                'name': 'Dr. Sneha Reddy',
                'specialization': 'Endocrinologist',
                'hospital': 'SRM Hospital',
                'experience_years': 10,
                'availability': 'Mon-Fri 11AM-7PM',
                'rating': 4.6,
                'consultation_fee': 900,
                'phone': '+91-9876543213',
                'email': 'sneha.reddy@apollo.com'
            },
            
            # Hepatologists
            {
                'doctor_id': 5,
                'name': 'Dr. Vikram Singh',
                'specialization': 'Hepatologist',
                'hospital': 'SRM Hospital',
                'experience_years': 20,
                'availability': 'Mon-Sat 8AM-2PM',
                'rating': 4.9,
                'consultation_fee': 2000,
                'phone': '+91-9876543214',
                'email': 'vikram.singh@medanta.com'
            },
            {
                'doctor_id': 6,
                'name': 'Dr. Anjali Mehta',
                'specialization': 'Hepatologist',
                'hospital': 'SRM Hospital',
                'experience_years': 14,
                'availability': 'Tue-Fri 10AM-5PM',
                'rating': 4.7,
                'consultation_fee': 1300,
                'phone': '+91-9876543215',
                'email': 'anjali.mehta@fortis.com'
            },
            
            # Nephrologists
            {
                'doctor_id': 7,
                'name': 'Dr. Suresh Iyer',
                'specialization': 'Nephrologist',
                'hospital': 'SRM Hospital',
                'experience_years': 16,
                'availability': 'Mon-Sat 9AM-5PM',
                'rating': 4.8,
                'consultation_fee': 1100,
                'phone': '+91-9876543216',
                'email': 'suresh.iyer@apollo.com'
            },
            {
                'doctor_id': 8,
                'name': 'Dr. Meera Nair',
                'specialization': 'Nephrologist',
                'hospital': 'SRM Hospital',
                'experience_years': 13,
                'availability': 'Tue-Sat 11AM-6PM',
                'rating': 4.6,
                'consultation_fee': 1000,
                'phone': '+91-9876543217',
                'email': 'meera.nair@max.com'
            },
            
            # General Physicians
            {
                'doctor_id': 9,
                'name': 'Dr. Ravi Verma',
                'specialization': 'General Physician',
                'hospital': 'SRM Hospital',
                'experience_years': 8,
                'availability': 'Mon-Sat 8AM-8PM',
                'rating': 4.5,
                'consultation_fee': 500,
                'phone': '+91-9876543218',
                'email': 'ravi.verma@cityhospital.com'
            },
            {
                'doctor_id': 10,
                'name': 'Dr. Kavita Desai',
                'specialization': 'General Physician',
                'hospital': 'SRM Hospital',
                'experience_years': 6,
                'availability': 'Mon-Fri 9AM-9PM',
                'rating': 4.4,
                'consultation_fee': 400,
                'phone': '+91-9876543219',
                'email': 'kavita.desai@healthplus.com'
            }
        ]
        
        self.doctors_df = pd.DataFrame(sample_doctors)

        self.doctors_df.to_csv(self.doctors_data_path, index=False)
        logger.info(f"Sample doctors database saved to {self.doctors_data_path}")
    
    def recommend_doctors(self, predictions: Dict, risk_levels: Dict, 
                         top_n: int = 3) -> Dict[str, List[Dict]]:
        """
        Recommend doctors based on disease predictions
        
        Args:
            predictions: Disease predictions from AI model
            risk_levels: Risk levels for each disease
            top_n: Number of doctors to recommend per disease
            
        Returns:
            Dictionary mapping disease to list of recommended doctors
        """
        recommendations = {}

        for disease, prediction_data in predictions.items():
            # Extract risk level - handle both dict and string formats
            if isinstance(prediction_data, dict):
                risk_level = prediction_data.get('risk_level', '').lower()
            else:
                risk_level = str(prediction_data).lower()
            
            # Only recommend for moderate and high risk
            if risk_level in ['moderate', 'high']:
                #Uses the exact disease key from predictions
                specialization = self.disease_specialization_map.get(disease)
                
                if specialization:
                    doctors = self.get_doctors_by_specialization(
                        specialization, top_n
                    )
                    if doctors:  # Only add if doctors are found
                        recommendations[disease] = doctors
                    logger.info(f"Recommending {len(doctors)} {specialization}s for {disease}")
        
        # If no specific recommendations, suggest general physician
        if not recommendations:
            general_doctors = self.get_doctors_by_specialization(
                'General Physician', 2
            )
            recommendations['general'] = general_doctors
            logger.info("No high/moderate risks found, recommending general physicians")
        
        return recommendations
    
    def get_doctors_by_specialization(self, specialization: str, 
                                     limit: int = 5) -> List[Dict]:
        """
        Get doctors by specialization, sorted by rating
        
        Args:
            specialization: Medical specialization
            limit: Maximum number of doctors to return
            
        Returns:
            List of doctor dictionaries
        """
        if self.doctors_df is None or self.doctors_df.empty:
            logger.warning("Doctors database is empty")
            return []
        
        # Filter by specialization
        filtered = self.doctors_df[
            self.doctors_df['specialization'] == specialization
        ]
        
        # Sort by rating (descending)
        sorted_doctors = filtered.sort_values('rating', ascending=False)
        
        # Limit results
        top_doctors = sorted_doctors.head(limit)
        
        # Convert to list of dictionaries
        return top_doctors.to_dict('records')
    
    def get_doctor_by_id(self, doctor_id: int) -> Optional[Dict]:
        """Get specific doctor by ID"""
        if self.doctors_df is None:
            return None
        
        doctor = self.doctors_df[self.doctors_df['doctor_id'] == doctor_id]
        
        if doctor.empty:
            return None
        
        return doctor.iloc[0].to_dict()
    
    def search_doctors(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search doctors by name, specialization, or hospital
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching doctors
        """
        if self.doctors_df is None or self.doctors_df.empty:
            return []
        
        query_lower = query.lower()
        
        # Search in name, specialization, and hospital columns
        mask = (
            self.doctors_df['name'].str.lower().str.contains(query_lower, na=False) |
            self.doctors_df['specialization'].str.lower().str.contains(query_lower, na=False) |
            self.doctors_df['hospital'].str.lower().str.contains(query_lower, na=False)
        )
        
        results = self.doctors_df[mask].sort_values('rating', ascending=False)
        
        return results.head(limit).to_dict('records')
    
    def get_all_specializations(self) -> List[str]:
        """Get list of all available specializations"""
        if self.doctors_df is None or self.doctors_df.empty:
            return []
        
        return sorted(self.doctors_df['specialization'].unique().tolist())
    
    def get_doctors_by_hospital(self, hospital: str) -> List[Dict]:
        """Get all doctors from a specific hospital"""
        if self.doctors_df is None or self.doctors_df.empty:
            return []
        
        filtered = self.doctors_df[self.doctors_df['hospital'] == hospital]
        return filtered.sort_values('rating', ascending=False).to_dict('records')
    
    def format_doctor_card(self, doctor: Dict) -> str:
        """
        Format doctor information as a readable card
        
        Args:
            doctor: Doctor dictionary
            
        Returns:
            Formatted string
        """
        card = f"""
╔════════════════════════════════════════════════════════
║ {doctor['name']}
║ {doctor['specialization']}
╠════════════════════════════════════════════════════════
║ 🏥 Hospital: {doctor['hospital']}
║ 💼 Experience: {doctor['experience_years']} years
║ ⭐ Rating: {doctor['rating']}/5.0
║ 💰 Consultation Fee: ₹{doctor['consultation_fee']}
║ 🕐 Availability: {doctor['availability']}
║ 📞 Phone: {doctor['phone']}
║ 📧 Email: {doctor['email']}
╚════════════════════════════════════════════════════════
"""
        return card
    
    def generate_recommendation_report(self, recommendations: Dict) -> str:
        """
        Generate a comprehensive recommendation report
        
        Args:
            recommendations: Disease to doctors mapping
            
        Returns:
            Formatted recommendation report
        """
        report = "\n" + "="*60 + "\n"
        report += "        DOCTOR RECOMMENDATION REPORT\n"
        report += "="*60 + "\n\n"
        
        if not recommendations:
            report += "No specific doctor recommendations at this time.\n"
            report += "Your health markers appear normal.\n"
            return report
        
        for disease, doctors in recommendations.items():
            if disease == 'general':
                report += "📋 GENERAL HEALTH CONSULTATION\n"
            else:
                report += f"⚠️  {disease.upper().replace('_', ' ')} CONCERN - Specialists Recommended\n"
            
            report += "-" * 60 + "\n"
            
            if not doctors:
                report += "No doctors available for this specialization.\n\n"
                continue
            
            for i, doctor in enumerate(doctors, 1):
                report += f"\n{i}. {doctor['name']}\n"
                report += f"   Specialization: {doctor['specialization']}\n"
                report += f"   Hospital: {doctor['hospital']}\n"
                report += f"   Experience: {doctor['experience_years']} years\n"
                report += f"   Rating: {doctor['rating']}⭐\n"
                report += f"   Fee: ₹{doctor['consultation_fee']}\n"
                report += f"   Availability: {doctor['availability']}\n"
                report += f"   Contact: {doctor['phone']}\n"
            
            report += "\n" + "-" * 60 + "\n\n"
        
        report += "="*60 + "\n"
        report += "Please book an appointment at your earliest convenience.\n"
        report += "="*60 + "\n"
        
        return report
    
    def get_statistics(self) -> Dict:
        """Get statistics about doctors database"""
        if self.doctors_df is None or self.doctors_df.empty:
            return {}
        
        stats = {
            'total_doctors': len(self.doctors_df),
            'specializations': len(self.doctors_df['specialization'].unique()),
            'hospitals': len(self.doctors_df['hospital'].unique()),
            'avg_experience': round(self.doctors_df['experience_years'].mean(), 1),
            'avg_rating': round(self.doctors_df['rating'].mean(), 2),
            'avg_fee': round(self.doctors_df['consultation_fee'].mean(), 0)
        }
        
        return stats