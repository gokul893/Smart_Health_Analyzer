"""
Agentic AI Health Analyzer with LLM Integration
Uses TinyLlama for intelligent health report summarization and runs
multiple disease prediction models.
"""

import os
import warnings
import joblib
import pandas as pd
import json
from typing import Dict
import torch #type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer #type: ignore

# Assumes you have 'doctor_recommendation.py' with this class
from doctor_recommendation import DoctorRecommendationSystem

# Suppress common warnings for a cleaner output
warnings.filterwarnings('ignore')


class AgenticAIModel:
    """
    An agentic AI system that orchestrates multiple tasks:
    1. Loads pre-trained disease prediction models.
    2. Uses an LLM to extract structured data from unstructured report text.
    3. Preprocesses data for each specific model.
    4. Generates an intelligent, human-readable summary of health risks.
    5. Provides doctor recommendations based on the analysis.
    """

    def __init__(self, models_dir: str = "Models"):
        """Initializes all components of the agentic health analyzer."""
        self.models_dir = models_dir
        self.models = {}
        self.preprocessors = {}
        
        # --- Initialization Sequence ---
        self._load_llm()
        self._load_prediction_models()
        
        print("\n👨‍⚕️  Initializing Doctor Recommendation System...")
        self.doctor_recommender = DoctorRecommendationSystem()
        
        print("\n✅ Agentic Health Analyzer is Ready!\n")

    def _load_llm(self):
        """Loads the TinyLlama language model with automatic CPU/GPU detection."""
        print("\n🧠 Initializing Language Model Agent (TinyLlama)...")
        try:
            # Using the smaller, more efficient TinyLlama model
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            if torch.cuda.is_available():
                print("   --> NVIDIA GPU detected. Loading model in optimized mode.")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto"
                )
            else:
                print("   --> No NVIDIA GPU detected. Loading model on CPU (this may be slower).")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True
                )
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print("   ✅ Language Model Loaded Successfully")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load LLM. Summaries will be basic. Error: {e}")
            self.llm_model = None
            self.tokenizer = None

    def _load_prediction_models(self):
        """Loads all 5 disease prediction models and their preprocessors."""
        print("\n📊 Loading Disease Prediction Models...")
        model_names = ['kidney_disease', 'heart_disease', 'diabetes', 'liver', 'thyroid']
        for name in model_names:
            try:
                model_path = os.path.join(self.models_dir, f'{name}_model.pkl')
                preprocessor_path = os.path.join(self.models_dir, f'{name}_preprocessing_data.pkl')
                self.models[name] = joblib.load(model_path)
                self.preprocessors[name] = joblib.load(preprocessor_path)
                print(f"   ✓ Loaded {name.replace('_', ' ').title()} model")
            except Exception as e:
                print(f"   ✗ Error loading {name} model: {e}")

    def extract_structured_data(self, report_text: str) -> Dict:
        """Uses the LLM to parse unstructured text into a data dictionary."""
        if not self.llm_model:
            print("LLM not available, cannot extract structured data.")
            return {}

        prompt = f"""
        You are a medical data extraction expert. Read the following medical report text and extract the key values for the prediction models.
        Extract only the following fields: 'age', 'sex' (0 for female, 1 for male), 'bmi', 'trestbps' (systolic blood pressure), 'chol' (total cholesterol), 'fbs' (fasting blood sugar > 120mg/dL is 1, else 0), 'restecg', 'thalach' (maximum heart rate), 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'blood_glucose_level', 'hba1c_level', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'.
        Provide the output as a clean JSON object and nothing else. If a value is not found, omit the key.

        REPORT TEXT:
        ---
        {report_text[:4000]}
        ---

        JSON OUTPUT:
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.llm_model.device)
            outputs = self.llm_model.generate(**inputs, max_new_tokens=512)
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            json_str = response_text.split("JSON OUTPUT:")[-1].strip()
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end != -1:
                clean_json_str = json_str[start:end]
                return json.loads(clean_json_str)
            return {}
        except Exception as e:
            print(f"Error extracting structured data from LLM response: {e}")
            return {}

    def predict_all_diseases(self, patient_data: Dict) -> Dict:
        """Runs predictions for all loaded disease models."""
        if not patient_data:
            return {name: {'prediction': 0, 'risk_score': 0.0, 'risk_level': 'Unknown', 'status': 'NoData'} for name in self.models.keys()}
        return {name: self._get_prediction(name, patient_data) for name in self.models.keys()}

    def _get_prediction(self, model_name: str, patient_data: Dict) -> Dict:
        """Generates a single prediction, including risk score and level."""
        try:
            model = self.models[model_name]
            df_processed = self._preprocess_input(patient_data, model_name)
            proba = model.predict_proba(df_processed)[0]
            prediction = int(proba.argmax())
            risk_score = float(proba[1])
            return {
                'prediction': prediction,
                'risk_score': risk_score,
                'risk_level': self._categorize_risk(risk_score),
                'status': 'positive' if prediction == 1 else 'negative'
            }
        except Exception as e:
            print(f"Error predicting for {model_name}: {e}")
            return {'prediction': 0, 'risk_score': 0.0, 'risk_level': 'Unknown', 'status': 'Error'}

    def _preprocess_input(self, data: Dict, model_name: str) -> pd.DataFrame:
        """Correctly preprocesses input data for a specific model."""
        df = pd.DataFrame([data])
        preprocessor = self.preprocessors[model_name]

        if model_name in ['kidney_disease', 'thyroid']:
            if 'encoders' in preprocessor:
                for col, le in preprocessor['encoders'].items():
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: le.transform([x])[0] if str(x) in le.classes_ else -1)
            if 'training_data_medians' in preprocessor:
                df.fillna(preprocessor['training_data_medians'], inplace=True)
        
        elif model_name == 'heart_disease':
            categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
            df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)
        
        elif model_name == 'diabetes':
            if 'training_data_medians' in preprocessor:
                df.fillna(preprocessor['training_data_medians'], inplace=True)

        elif model_name == 'liver':
            imputation_values = preprocessor.get('imputation_values', {})
            df.fillna(imputation_values, inplace=True)

        return df.reindex(columns=preprocessor['column_order'], fill_value=0)

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorizes a numeric risk score into a readable level."""
        if risk_score >= 0.75: return 'High'
        if risk_score >= 0.40: return 'Moderate'
        return 'Low'

    def generate_llm_summary(self, predictions: Dict) -> str:
        """Uses the LLM to generate an intelligent, structured health summary."""
        if not self.llm_model:
            return self._generate_basic_summary(predictions)
        
        context = "Medical Health Analysis Report:\n"
        for disease, result in predictions.items():
            if result['status'] != 'Error':
                context += f"- {disease.replace('_', ' ').title()}: {result['risk_level']} risk ({result['risk_score']*100:.1f}%)\n"
        
        prompt = f"""You are an expert AI health assistant. Based on the following risk analysis, provide a concise summary in three sections:
1.  **KEY CONCERNS**: List conditions with 'High' or 'Moderate' risk.
2.  **POSITIVE INDICATORS**: Mention conditions with 'Low' risk.
3.  **OVERALL RECOMMENDATION**: Give general advice, not a diagnosis.

Keep the response under 150 words and end with a disclaimer to consult a doctor.

{context}
Summary:"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            outputs = self.llm_model.generate(**inputs, max_new_tokens=200, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Summary:")[-1].strip()
        except Exception as e:
            print(f"Error generating LLM summary: {e}")
            return self._generate_basic_summary(predictions)

    def _generate_basic_summary(self, predictions: Dict) -> str:
        """A fallback summary generator for when the LLM is unavailable."""
        summary = "Health Analysis Summary:\n"
        high_risks = [k.replace('_', ' ').title() for k, v in predictions.items() if v.get('risk_level') == 'High']
        if high_risks: summary += f"- High risk detected for: {', '.join(high_risks)}. Immediate consultation is advised.\n"
        moderate_risks = [k.replace('_', ' ').title() for k, v in predictions.items() if v.get('risk_level') == 'Moderate']
        if moderate_risks: summary += f"- Moderate risk for: {', '.join(moderate_risks)}. Please schedule a checkup.\n"
        low_risks = [k.replace('_', ' ').title() for k, v in predictions.items() if v.get('risk_level') == 'Low']
        if low_risks: summary += f"- Low risk noted for: {', '.join(low_risks)}. Continue monitoring health.\n"
        return summary

    def get_doctor_recommendations(self, predictions: Dict) -> Dict:
        """Gets tailored doctor recommendations based on high or moderate risk."""
        risk_levels = {disease: result['risk_level'].lower() for disease, result in predictions.items()}
        return self.doctor_recommender.recommend_doctors(predictions, risk_levels)
    
    def chat(self, user_input: str, context: str = "") -> str:
        """
        Handles chat interactions for the AI Chat page.
        """
        if not self.llm_model:
            return "I'm sorry, the AI chat model is not available at the moment."
        
        prompt = f"""
        You are a helpful AI health assistant. A user is asking a question. Use the provided context from their last health report analysis if it is relevant. Do not give medical advice. Always encourage consulting a doctor for serious concerns.

        CONTEXT FROM LAST ANALYSIS:
        {context}

        USER QUESTION:
        {user_input}

        ANSWER:
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            outputs = self.llm_model.generate(**inputs, max_new_tokens=250, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("ANSWER:")[-1].strip()
        except Exception as e:
            print(f"Error during chat generation: {e}")
            return "I apologize, I encountered an error and cannot respond right now."