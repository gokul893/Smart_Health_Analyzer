# 🏥 Smart Health Report Analyzer with Doctor Recommendation

An AI-powered healthcare system that analyzes medical reports, predicts disease risks, and recommends appropriate doctors using Agentic AI and Machine Learning.

## 📋 Overview

This system leverages cutting-edge AI technology to:
- **Analyze Health Reports**: Extract text from PDF, images, and documents
- **Predict Diseases**: Assess risk for Heart Disease, Diabetes, Liver Disease, Kidney Disease, and Thyroid Disorders
- **Recommend Doctors**: Match patients with appropriate specialists based on predictions
- **Manage Appointments**: Schedule, reschedule, and track medical appointments
- **Interactive AI Chat**: Conversational AI assistant for health queries

## 🚀 Features

### Core Functionality
- ✅ Multi-format report processing (PDF, Images, Text, DOCX)
- ✅ 5 Disease prediction models (Heart, Diabetes, Liver, Kidney, Thyroid)
- ✅ AI-powered report analysis with explanations
- ✅ Automated doctor recommendations based on risk assessment
- ✅ Complete appointment management system
- ✅ Patient data management with secure authentication
- ✅ Interactive AI chatbot for health queries
- ✅ Web-based user interface (Streamlit)
- ✅ RESTful API backend (FastAPI)

### Technical Highlights
- **No Paid APIs**: Uses free Hugging Face models
- **Containerized**: Docker support for easy deployment
- **Scalable Architecture**: Modular design for easy extension
- **Database**: MySQL for reliable data storage
- **OCR Support**: Extract text from images using Tesseract

## 📁 Project Structure

```
Smart_Health_Analyzer/
│
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── README.md                     # This file
│
├── main.py                       # FastAPI backend server
├── interface.py                  # Streamlit frontend
│
├── database_system.py            # MySQL database management
├── appointment_system.py         # Appointment scheduling
├── prescription_analysis.py      # Document text extraction
├── agentic_ai_model.py          # AI disease prediction models
├── doctor_recommendation.py      # Doctor matching system
│
├── models/                       # Trained ML models (auto-generated)
├── uploads/                      # Temporary file storage
├── logs/                         # Application logs
└── doctors_database.csv          # Doctor information database
```

## 🛠️ Technology Stack

### Backend
- **Python 3.11+**
- **FastAPI**: Modern web framework for APIs
- **MySQL**: Relational database
- **SQLAlchemy**: Database ORM

### Frontend
- **Streamlit**: Interactive web interface
- **Gradio**: Alternative UI option

### AI/ML
- **Hugging Face Transformers**: Free conversational AI models
- **scikit-learn**: Machine learning models
- **PyTorch**: Deep learning framework

### Document Processing
- **PyMuPDF**: PDF text extraction
- **pytesseract**: OCR for images
- **Pillow**: Image processing
- **python-docx**: DOCX processing

### Containerization
- **Docker**: Application containerization

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- MySQL Server 8.0+
- Tesseract OCR
- Docker (optional)

### Method 1: Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smart-health-analyzer.git
cd smart-health-analyzer
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

4. **Setup MySQL Database**

Create a MySQL database:
```sql
CREATE DATABASE health_analyzer;
CREATE USER 'health_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON health_analyzer.* TO 'health_user'@'localhost';
FLUSH PRIVILEGES;
```

5. **Configure Database Connection**

Edit `database_system.py` and update connection parameters:
```python
DatabaseSystem(
    host="localhost",
    user="health_user",
    password="your_password",
    database="health_analyzer"
)
```

6. **Run the application**

Terminal 1 - Start FastAPI backend:
```bash
python main.py
```

Terminal 2 - Start Streamlit frontend:
```bash
streamlit run interface.py
```

7. **Access the application**
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs

### Method 2: Docker Setup

1. **Build Docker image**
```bash
docker build -t smart-health-analyzer .
```

2. **Run MySQL container**
```bash
docker run --name mysql-health \
  -e MYSQL_ROOT_PASSWORD=rootpassword \
  -e MYSQL_DATABASE=health_analyzer \
  -e MYSQL_USER=health_user \
  -e MYSQL_PASSWORD=password \
  -p 3306:3306 \
  -d mysql:8.0
```

3. **Run application container**
```bash
docker run -p 8000:8000 -p 8501:8501 \
  --link mysql-health:mysql \
  -e DB_HOST=mysql \
  -e DB_USER=health_user \
  -e DB_PASSWORD=password \
  smart-health-analyzer
```

4. **Access the application**
- Frontend: http://localhost:8501
- API: http://localhost:8000

## 📖 Usage Guide

### 1. Patient Registration
- Navigate to the registration tab
- Fill in your details (Login ID, Password, Name, etc.)
- Click "Register"

### 2. Login
- Enter your Login ID and Password
- Click "Login"

### 3. Upload Health Report
- Go to "Upload Report" page
- Select your medical report file (PDF, Image, Text, or DOCX)
- Click "Analyze Report"
- Wait for AI analysis to complete

### 4. View Analysis Results
- Review overall health summary
- Check disease risk assessments
- Read medical recommendations
- View recommended doctors

### 5. Book Appointment
- Go to "Appointments" page
- Select a recommended doctor or browse all doctors
- Choose date and time
- Provide purpose of visit
- Click "Book Appointment"

### 6. Manage Appointments
- View all your appointments in "My Appointments"
- Cancel or reschedule as needed

### 7. Chat with AI
- Go to "AI Chat" page
- Ask questions about your health or reports
- Get instant AI-powered responses

## 🔌 API Endpoints

### Authentication
- `POST /api/register` - Register new patient
- `POST /api/login` - Patient login

### Report Analysis
- `POST /api/analyze` - Upload and analyze health report

### Doctors
- `GET /api/doctors` - Get list of doctors
- `GET /api/doctors/search?query=` - Search doctors

### Appointments
- `POST /api/appointments` - Create appointment
- `GET /api/appointments/{patient_id}` - Get patient appointments
- `DELETE /api/appointments/{appointment_id}` - Cancel appointment

### AI Chat
- `POST /api/chat` - Chat with AI assistant

### System
- `GET /health` - Health check
- `GET /api/statistics` - System statistics

## 🧪 Testing

Run API tests:
```bash
pytest tests/
```

Test individual components:
```bash
python -m pytest tests/test_database.py
python -m pytest tests/test_ai_model.py
```

## 🔒 Security Considerations

**Important**: This is a demonstration project. For production use:

1. **Password Security**: Implement proper password hashing (bcrypt, argon2)
2. **API Authentication**: Add JWT tokens or OAuth2
3. **HTTPS**: Use SSL/TLS certificates
4. **Input Validation**: Sanitize all user inputs
5. **File Upload Security**: Validate and scan uploaded files
6. **Database Security**: Use environment variables for credentials
7. **Rate Limiting**: Implement API rate limiting
8. **HIPAA Compliance**: Ensure medical data compliance

## 🐛 Troubleshooting

### Issue: Models not training
**Solution**: Ensure sufficient disk space and RAM. Training may take 2-5 minutes.

### Issue: OCR not working
**Solution**: Install Tesseract OCR and set the path in `prescription_analysis.py`

### Issue: Database connection failed
**Solution**: Verify MySQL is running and credentials are correct

### Issue: Port already in use
**Solution**: Change ports in `main.py` (FastAPI) or run Streamlit with:
```bash
streamlit run interface.py --server.port=8502
```

## 📊 Sample Data

The system includes sample doctors database with:
- 10 doctors across 5 specializations
- Cardiologists, Endocrinologists, Hepatologists, Nephrologists, General Physicians
- Hospital affiliations, ratings, fees, and availability

## 🔮 Future Enhancements

- [ ] Integration with real medical datasets
- [ ] Support for more diseases and conditions
- [ ] Telemedicine video consultation
- [ ] Mobile application (React Native)
- [ ] Electronic Health Records (EHR) integration
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Insurance claim integration
- [ ] Prescription management
- [ ] Lab test tracking

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## ⚠️ Disclaimer

**Medical Disclaimer**: This system is for educational and research purposes only. It does NOT replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

## 👥 Authors

- Development Team
- Contact: support@healthanalyzer.com

## 🙏 Acknowledgments

- Hugging Face for free AI models
- scikit-learn community
- Streamlit team
- FastAPI contributors
- Open-source medical research community

## 📞 Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/smart-health-analyzer/issues
- Email: support@healthanalyzer.com
- Documentation: https://docs.healthanalyzer.com

---

**Made with ❤️ for better healthcare accessibility**