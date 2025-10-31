"""
Streamlit Web Interface for Smart Health Report Analyzer
Provides user-friendly frontend for report analysis and doctor recommendations
"""

import streamlit as st # type: ignore
from datetime import datetime, date, time
import pandas as pd
import hashlib

# Import custom modules for direct use
from database_system import get_database
from appointment_system import AppointmentSystem
from prescription_analysis import PrescriptionAnalyzer
from agentic_ai_model import AgenticAIModel
from doctor_recommendation import DoctorRecommendationSystem

# Page configuration
st.set_page_config(
    page_title="Smart Health Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem; color: #1f77b4; text-align: center; padding: 1rem;
    }
    .success-box, .warning-box, .danger-box {
        padding: 1rem; border-radius: 0.5rem; border: 1px solid;
    }
    .success-box { background-color: #d4edda; border-color: #c3e6cb; color: #155724; }
    .warning-box { background-color: #fff3cd; border-color: #ffeeba; color: #856404; }
    .danger-box { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all system components once and cache them."""
    return (
        get_database(),
        AppointmentSystem(),
        PrescriptionAnalyzer(),
        AgenticAIModel(),
        DoctorRecommendationSystem()
    )

try:
    db, appt_sys, presc_analyzer, ai_model, doc_recommender = initialize_components()
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Added session state to control the current page view
if 'page' not in st.session_state:
    st.session_state.page = "Upload Report"


def login_page():
    """Login and registration page"""
    st.markdown("<h1 class='main-header'>🏥 Smart Health Report Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("### AI-Powered Health Analysis with Doctor Recommendations")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Patient Login")
        with st.form("login_form"):
            login_id = st.text_input("Login ID")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")

            if submitted:
                if login_id and password:
                    password_hash = hashlib.sha256(password.encode()).hexdigest()
                    patient = db.authenticate_patient(login_id, password_hash)
                    if patient:
                        st.session_state.logged_in = True
                        st.session_state.patient_data = patient
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                else:
                    st.warning("Please enter both login ID and password")
    
    with tab2:
        st.subheader("New Patient Registration")
        with st.form("registration_form"):
            col1, col2 = st.columns(2)
            with col1:
                reg_login_id = st.text_input("Choose Login ID*")
                reg_password = st.text_input("Choose Password*", type="password")
                patient_name = st.text_input("Full Name*")
                age = st.number_input("Age", min_value=0, max_value=120, step=1)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            with col2:
                weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
                height = st.number_input("Height (cm)", min_value=0.0, step=0.1)
                phone = st.text_input("Phone Number")
                email = st.text_input("Email Address")
            
            submitted = st.form_submit_button("Register", type="primary")
            if submitted:
                if reg_login_id and reg_password and patient_name:
                    password_hash = hashlib.sha256(reg_password.encode()).hexdigest()
                    patient_id = db.register_patient(
                        login_id=reg_login_id, password_hash=password_hash, patient_name=patient_name,
                        age=age or None, weight=weight or None, height=height or None,
                        gender=gender, phone=phone or None, email=email or None
                    )
                    if patient_id:
                        st.success("Registration successful! Please go to the Login tab.")
                    else:
                        st.error("Registration failed. Login ID may already exist.")
                else:
                    st.error("Please fill all required fields (*)")


def dashboard_page():
    """Main dashboard after login"""
    st.sidebar.title(f"Welcome, {st.session_state.patient_data['patient_name']}!")

    # The radio button now controls the 'page' session state variable
    page_options = ["Upload Report", "My Past Reports", "Find Doctors", "My Appointments", "AI Chat"]
    st.session_state.page = st.sidebar.radio(
        "Navigation",
        page_options,
        index=page_options.index(st.session_state.page)
    )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        # Clear all session data on logout for a clean state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Page rendering is now based on the session state
    if st.session_state.page == "Upload Report":
        upload_report_page()
    elif st.session_state.page == "My Past Reports":
        my_reports_page()
    elif st.session_state.page == "Find Doctors":
        doctors_page()
    elif st.session_state.page == "My Appointments":
        appointments_page()
    elif st.session_state.page == "AI Chat":
        ai_chat_page()


def upload_report_page():
    # This function remains unchanged
    st.title("📄 Upload & Analyze Health Report")
    st.markdown("Upload a medical report file to get an AI-powered analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a report file", type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'docx']
    )
    
    if uploaded_file:
        if st.button("Analyze Report", type="primary", use_container_width=True):
            with st.spinner("Analyzing your report... This may take a moment."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    extracted_text, file_type = presc_analyzer.process_file_from_bytes(file_bytes, uploaded_file.name)
                    if not extracted_text:
                        st.error("Could not extract text from the file.")
                        return

                    patient_data_from_report = presc_analyzer.extract_values_with_regex(extracted_text)
                    predictions = ai_model.predict_all_diseases(patient_data_from_report)
                    llm_summary = ai_model.generate_llm_summary(predictions)
                    doctor_recs = ai_model.get_doctor_recommendations(predictions)

                    st.session_state.analysis_results = {
                        'predictions': predictions,
                        'llm_summary': llm_summary,
                        'doctor_recommendations': doctor_recs
                    }
                    st.success("Analysis Complete!")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

    if st.session_state.analysis_results:
        display_analysis_results()


def display_analysis_results():
    """Display analysis results with a functional booking button."""
    results = st.session_state.analysis_results
    st.markdown("---")
    st.header("📊 Analysis Results")

    st.subheader("🧠 AI-Generated Health Summary")
    st.markdown(f"<div class='info-box'>{results['llm_summary']}</div>", unsafe_allow_html=True)

    st.subheader("🩺 Disease Risk Assessment")
    predictions = results['predictions']
    cols = st.columns(len(predictions))
    for col, (disease, result) in zip(cols, predictions.items()):
        risk_level = result.get('risk_level', 'Unknown')
        risk_score = result.get('risk_score', 0) * 100
        col.metric(label=disease.replace('_', ' ').title(), value=f"{risk_score:.1f}%", delta=risk_level)

    st.subheader("👨‍⚕️ Recommended Doctor Consultations")
    doctor_recs = results['doctor_recommendations']
    if not any(doctor_recs.values()):
        st.info("No specific urgent consultations are recommended at this time.")
    else:
        for disease, doctors in doctor_recs.items():
            if doctors:
                st.markdown(f"**For {disease.replace('_', ' ').title()} Concerns:**")
                for doc in doctors:
                    with st.container(border=True):
                        st.write(f"**Dr. {doc['name']}** ({doc['specialization']}) at {doc['hospital']}")
                        # This button now changes the page
                        if st.button(f"Book with Dr. {doc['name']}", key=f"book_res_{doc['doctor_id']}"):
                            st.session_state.selected_doctor = doc
                            st.session_state.page = "My Appointments"
                            st.rerun()


def my_reports_page():
    st.title("📋 My Past Medical Reports")
    patient_id = st.session_state.patient_data['patient_id']
    reports = db.get_patient_reports(patient_id)
    if not reports:
        st.info("No reports found. Upload your first report.")
        return
    for report in reports:
        with st.expander(f"Report from {report['uploaded_at'].strftime('%Y-%m-%d %H:%M')}"):
            st.write(f"**Overall Risk Identified:** {report['risk_level']}")
            st.text_area("Report Text Preview", report['report_text'][:500] + "...", height=100, disabled=True)


def doctors_page():
    """Browse and search doctors with a functional booking button."""
    st.title("👨‍⚕️ Find Doctors")
    search_query = st.text_input("Search doctors by name, specialization, or hospital")
    
    doctors = doc_recommender.search_doctors(search_query) if search_query else doc_recommender.doctors_df.to_dict('records')
    
    if not doctors:
        st.info("No doctors found.")
    else:
        for doc in doctors:
            with st.container(border=True):
                st.write(f"**Dr. {doc['name']}** - {doc['specialization']} at {doc['hospital']}")
                st.write(f"Rating: {doc['rating']}⭐ | Fee: ₹{doc['consultation_fee']}")
                # This button now changes the page
                if st.button(f"Book Appointment", key=f"book_doc_{doc['doctor_id']}"):
                    st.session_state.selected_doctor = doc
                    st.session_state.page = "My Appointments"
                    st.rerun()


def appointments_page():
    # This function remains unchanged
    st.title("📅 My Appointments")
    
    tab1, tab2 = st.tabs(["Book New Appointment", "View My Appointments"])
    
    with tab1:
        st.subheader("Schedule an Appointment")
        with st.form("appointment_form"):
            selected_doctor = st.session_state.get('selected_doctor')
            if selected_doctor:
                st.success(f"Booking for Dr. {selected_doctor['name']}.")
                doctor_name = st.text_input("Doctor Name", value=selected_doctor['name'], disabled=True)
            else:
                st.info("You can select a doctor from the 'Find Doctors' page or from your analysis results.")
                doctor_name = st.text_input("Doctor Name*")

            appt_date = st.date_input("Appointment Date", min_value=date.today())
            appt_time = st.time_input("Appointment Time")
            purpose = st.text_area("Purpose of Visit (Optional)")
            
            submitted = st.form_submit_button("Confirm Booking", type="primary")
            if submitted and doctor_name:
                appointment_id = appt_sys.create_appointment(
                    patient_id=st.session_state.patient_data['patient_id'],
                    doctor_name=doctor_name,
                    appointment_date=appt_date,
                    appointment_time=appt_time,
                    purpose=purpose,
                    specialization=selected_doctor.get('specialization') if selected_doctor else None,
                    hospital=selected_doctor.get('hospital') if selected_doctor else None
                )
                if appointment_id:
                    st.success(f"✅ Appointment booked successfully! Your ID is {appointment_id}.")
                    if 'selected_doctor' in st.session_state:
                        del st.session_state['selected_doctor']
                else:
                    st.error("Booking failed. The time slot may be unavailable.")

    with tab2:
        st.subheader("Your Scheduled Appointments")
        patient_id = st.session_state.patient_data['patient_id']
        appointments = appt_sys.get_patient_appointments(patient_id)
        if not appointments:
            st.info("You have no scheduled appointments.")
        else:
            for appt in appointments:
                if appt['status'] == 'scheduled':
                    st.markdown(f"**Dr. {appt['doctor_name']}** on **{appt['appointment_date']}** at **{appt['appointment_time']}**")


def ai_chat_page():
    # This function remains unchanged
    st.title("💬 Chat with AI Health Assistant")
    st.info("Ask general health questions or about your last analysis.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = str(st.session_state.analysis_results) if st.session_state.analysis_results else ""
                response = ai_model.chat(prompt, context)
                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# Main app logic
def main():
    """Main function to control page navigation."""
    if not st.session_state.get('logged_in'):
        login_page()
    else:
        dashboard_page()

if __name__ == "__main__":
    main()