"""
Main Entry Point for Smart Health Report Analyzer
Integrates all modules and provides FastAPI backend
"""

import os
import logging
from datetime import datetime
from typing import Optional
import uvicorn # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException, Form # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from pydantic import BaseModel

# Import custom modules
from database_system import get_database, DatabaseSystem
from appointment_system import AppointmentSystem
from prescription_analysis import PrescriptionAnalyzer
from agentic_ai_model import AgenticAIModel
from doctor_recommendation import DoctorRecommendationSystem

# Create logs and uploads directories first
os.makedirs('logs', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Health Report Analyzer",
    description="AI-powered health report analysis with doctor recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system components
database = None
appointment_system = None
prescription_analyzer = None
ai_model = None
doctor_recommender = None


# Pydantic models for API requests
class PatientRegistration(BaseModel):
    login_id: str
    password: str
    patient_name: str
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class PatientLogin(BaseModel):
    login_id: str
    password: str


class AppointmentCreate(BaseModel):
    patient_id: int
    doctor_name: str
    appointment_date: str
    appointment_time: str
    specialization: Optional[str] = None
    purpose: Optional[str] = None
    hospital: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[int] = None
    context: Optional[str] = ""


@app.on_event("startup")
async def startup_event():
    """Initialize all system components on startup"""
    global database, appointment_system, prescription_analyzer, ai_model, doctor_recommender
    
    logger.info("Starting Smart Health Report Analyzer...")
    
    try:
        # Initialize database
        database = get_database()
        logger.info("✓ Database initialized")
        
        # Initialize appointment system
        appointment_system = AppointmentSystem()
        logger.info("✓ Appointment system initialized")
        
        # Initialize prescription analyzer
        prescription_analyzer = PrescriptionAnalyzer()
        logger.info("✓ Prescription analyzer initialized")
        
        # Initialize AI model (it will load the pre-trained models from disk)
        ai_model = AgenticAIModel() 
        logger.info("✓ AI model initialized")
        
        # Initialize doctor recommender
        doctor_recommender = DoctorRecommendationSystem()
        logger.info("✓ Doctor recommendation system initialized")
        
        logger.info("="*60)
        logger.info(" Smart Health Report Analyzer is ready!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Smart Health Report Analyzer...")
    if database:
        database.close()


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Smart Health Report Analyzer API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "register": "/api/register",
            "login": "/api/login",
            "analyze": "/api/analyze",
            "doctors": "/api/doctors",
            "appointments": "/api/appointments"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": database is not None,
            "ai_model": ai_model is not None,
            "appointment_system": appointment_system is not None
        }
    }


@app.post("/api/register")
async def register_patient(patient: PatientRegistration):
    """Register a new patient"""
    try:
        # Simple password hashing (use proper hashing in production)
        import hashlib
        password_hash = hashlib.sha256(patient.password.encode()).hexdigest()
        
        patient_id = database.register_patient(
            login_id=patient.login_id,
            password_hash=password_hash,
            patient_name=patient.patient_name,
            age=patient.age,
            weight=patient.weight,
            height=patient.height,
            gender=patient.gender,
            phone=patient.phone,
            email=patient.email
        )
        
        if patient_id:
            return {
                "success": True,
                "patient_id": patient_id,
                "message": "Patient registered successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Registration failed")
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/login")
async def login_patient(credentials: PatientLogin):
    """Authenticate patient login"""
    try:
        import hashlib
        password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
        
        patient = database.authenticate_patient(
            login_id=credentials.login_id,
            password_hash=password_hash
        )
        
        if patient:
            # Remove sensitive data
            patient.pop('password_hash', None)
            return {
                "success": True,
                "patient": patient,
                "message": "Login successful"
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_report(
    file: UploadFile = File(...),
    patient_id: int = Form(...)
):
    """Analyze uploaded health report"""
    try:
        # Save uploaded file temporarily
        file_path = f"uploads/{file.filename}"
        content = await file.read()
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Extract text from file
        extracted_text, file_type = prescription_analyzer.process_file(file_path)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Validate report
        is_valid, validation_msg = prescription_analyzer.validate_report(extracted_text)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_msg)
        
        # Analyze with AI model
        analysis_results = ai_model.analyze_report(extracted_text)
        
        # Get doctor recommendations
        doctor_recommendations = doctor_recommender.recommend_doctors(
            analysis_results['predictions'],
            analysis_results['risk_levels']
        )
        
        # Generate recommendation report
        recommendation_report = doctor_recommender.generate_recommendation_report(
            doctor_recommendations
        )
        
        # Save to database
        database.save_medical_report(
            patient_id=patient_id,
            report_type=file_type,
            report_text=extracted_text[:5000],  # Truncate for storage
            predictions=analysis_results['predictions'],
            risk_level=max(analysis_results['risk_levels'].values(), 
                          key=lambda x: ['low', 'moderate', 'high', 'unknown'].index(x))
        )
        
        # Clean up temporary file
        os.remove(file_path)
        
        return {
            "success": True,
            "analysis": analysis_results,
            "doctors": doctor_recommendations,
            "recommendation_report": recommendation_report,
            "message": "Report analyzed successfully"
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/doctors")
async def get_doctors(specialization: Optional[str] = None, limit: int = 10):
    """Get list of doctors"""
    try:
        if specialization:
            doctors = doctor_recommender.get_doctors_by_specialization(
                specialization, limit
            )
        else:
            # Return all doctors
            if doctor_recommender.doctors_df is not None:
                doctors = doctor_recommender.doctors_df.head(limit).to_dict('records')
            else:
                doctors = []
        
        return {
            "success": True,
            "doctors": doctors,
            "count": len(doctors)
        }
        
    except Exception as e:
        logger.error(f"Error fetching doctors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/doctors/search")
async def search_doctors(query: str, limit: int = 10):
    """Search doctors by name, specialization, or hospital"""
    try:
        doctors = doctor_recommender.search_doctors(query, limit)
        
        return {
            "success": True,
            "doctors": doctors,
            "count": len(doctors)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/appointments")
async def create_appointment(appointment: AppointmentCreate):
    """Create a new appointment"""
    try:
        from datetime import datetime
        
        # Parse date and time
        appt_date = datetime.strptime(appointment.appointment_date, "%Y-%m-%d").date()
        appt_time = datetime.strptime(appointment.appointment_time, "%H:%M").time()
        
        appointment_id = appointment_system.create_appointment(
            patient_id=appointment.patient_id,
            doctor_name=appointment.doctor_name,
            appointment_date=appt_date,
            appointment_time=appt_time,
            specialization=appointment.specialization,
            purpose=appointment.purpose,
            hospital=appointment.hospital
        )
        
        if appointment_id:
            return {
                "success": True,
                "appointment_id": appointment_id,
                "message": "Appointment created successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Could not create appointment")
            
    except Exception as e:
        logger.error(f"Appointment creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/appointments/{patient_id}")
async def get_patient_appointments(patient_id: int):
    """Get all appointments for a patient"""
    try:
        appointments = appointment_system.get_patient_appointments(patient_id)
        
        return {
            "success": True,
            "appointments": appointments,
            "count": len(appointments)
        }
        
    except Exception as e:
        logger.error(f"Error fetching appointments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/appointments/{appointment_id}")
async def cancel_appointment(appointment_id: int):
    """Cancel an appointment"""
    try:
        success = appointment_system.cancel_appointment(appointment_id)
        
        if success:
            return {
                "success": True,
                "message": "Appointment cancelled successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Could not cancel appointment")
            
    except Exception as e:
        logger.error(f"Error cancelling appointment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """Chat with AI assistant about health reports"""
    try:
        response = ai_model.chat(request.message, request.context)
        
        return {
            "success": True,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics"""
    try:
        doctor_stats = doctor_recommender.get_statistics()
        
        return {
            "success": True,
            "statistics": doctor_stats
        }
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run FastAPI server
    uvicorn.run(
        "health_analyzer_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )