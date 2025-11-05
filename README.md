# Smart Health Report Analyzer

An intelligent, patient-centric healthcare assistant designed to analyze complex medical reports, predict disease risks using machine learning, and provide actionable recommendations, including specialist consultations. This project leverages Agentic AI to bridge the gap between complex medical data and patient understanding.

## 📝 Problem Statement

Patients frequently receive diagnostic health reports containing complex medical data and terminology that are difficult for them to interpret without expert medical knowledge. This lack of immediate understanding can lead to significant anxiety, delays in seeking appropriate care, and confusion about necessary next steps. Existing digital tools often provide only generic risk scores without offering personalized insights or guidance on consulting the correct medical specialist.

## 💡 Solution

This project provides an intelligent, user-friendly system built to empower patients. The web application allows users to upload their health reports and receive an immediate, multi-faceted analysis.

The system's core functionality includes:
* **Analyzing** complex health reports from various file formats.
* **Predicting** potential risks for five chronic conditions: Diabetes, Heart Disease, Liver Disease, Kidney Disease, and Thyroid Disorder.
* **Summarizing** these findings in simple, easy-to-understand language using a Large Language Model (LLM).
* **Recommending** the correct type of medical specialist based on the analysis.
* **Facilitating** patient action through an integrated appointment booking system and an AI-powered chat.

## 🚀 Key Features

* **User Authentication:** Secure user registration and login system.
* **Multi-Format Report Upload:** Supports various file types, including PDF, JPG, PNG, DOCX, and TXT.
* **Text & Data Extraction:** Automatically processes uploaded files to extract both raw text and specific medical parameters (e.g., glucose, cholesterol, TSH) using Regex.
* **ML-Based Disease Prediction:** Utilizes five pre-trained Scikit-learn models to assess disease risks.
* **AI-Generated Summaries:** Employs an Agentic AI (LLM) to generate concise, human-readable summaries of the analysis results.
* **Specialist Recommendation:** Intelligently maps identified risks (e.g., 'Heart Disease') to the appropriate specialists (e.g., 'Cardiologist').
* **Appointment Management:** Allows users to browse recommended doctors, book new appointments, and view their scheduled appointments.
* **Conversational AI Chat:** Provides an interactive chat interface for users to ask general health questions or receive guided assistance with tasks like booking appointments.

## 🛠️ Technology Stack

The system is built entirely in Python, integrating several key libraries and frameworks.

| **Language** : Python 3.8+ - Core programming language |
| **Web Framework** : Streamlit - Building the interactive user interface |
| **Machine Learning** : Scikit-learn - Running the five disease prediction models |
| **Data Handling** : Pandas, NumPy - Data manipulation and preprocessing |
| **AI / NLP** : Hugging Face Transformers - Powering the LLM for summaries and chat |
| **Database** : MySQL - Storing user, appointment, and report data |
| **Data Extraction** : PyMuPDF (fitz), Pytesseract, Pillow, python-docx - Extracting text from PDF, image, and DOCX files |

## 🏗️ System Architecture

The application uses a modular, three-tier architecture.

1.  **Front-End (Presentation Tier):** A Streamlit web application (`health_analyzer_interface.py`) provides the UI for all user interactions.
2.  **Back-End (Logic Tier):** A set of Python modules handles all business logic:
    * `prescription_analysis.py`: Manages file ingestion, text extraction (including OCR), and Regex-based data parsing.
    * `agentic_ai_model.py`: The central "brain" that loads the LLM and ML models, orchestrates the prediction pipeline, generates summaries, and manages the AI chat.
    * `doctor_recommendation.py`: Loads a `doctors_database.csv` and contains the logic to map diseases to specialists.
    * `appointment_system.py`: Manages the logic for creating, viewing, and checking conflicts for appointments.
    * `database_system.py`: A dedicated module to handle all MySQL database connections and queries, featuring a robust `reconnect()` method.
3.  **Data Tier:**
    * **MySQL Database:** Stores persistent data for `patient_data`, `appointment_data`, and `medical_reports`.
    * **Pre-trained Models:** Saved `.pkl` files for the five Scikit-learn disease models.
    * **Doctor Database:** A local `.csv` file containing specialist information.

## ⚠️ Limitations

* **Informational Tool Only:** The system is an informational tool and is **not a substitute for professional medical diagnosis or advice**.
* **Limited Disease Scope:** Predictions are limited to the five specified diseases (Diabetes, Heart, Kidney, Liver, Thyroid).
* **Extraction Dependency:** The accuracy of the analysis heavily depends on the clarity and format of the uploaded report. Unstructured reports may lead to extraction errors.
* **Static Doctor Data:** The doctor recommendation module uses a limited, sample database and does not reflect real-time availability or a comprehensive directory of all specialists.

## 🔮 Future Enhancements

* **Expanded Disease Coverage:** Incorporate more ML models to predict a wider range of health conditions.
* **Advanced Data Extraction:** Move beyond Regex to use fine-tuned NLP models for more robust Named Entity Recognition (NER) from unstructured text.
* **Improved Recommendations:** Enhance the recommendation logic to include factors like user location, doctor ratings, and real-time availability.
* **Wearable Data Integration:** Connect to data streams from wearable devices (e.g., smartwatches) for continuous health monitoring.
