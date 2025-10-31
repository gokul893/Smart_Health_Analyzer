"""
Prescription and Medical Report Analysis Module
Handles text extraction from various file types and uses Regex for fast, structured data extraction.
"""

import os
import io
import re
from PIL import Image
import pytesseract # type: ignore
import fitz  # PyMuPDF
from docx import Document # type: ignore
from typing import Tuple, Dict

class PrescriptionAnalyzer:
    """Extracts and validates text from medical report files."""

    def __init__(self):
        pass

    def extract_values_with_regex(self, text: str) -> Dict:
        """
        Uses Regular Expressions (Regex) to quickly find and extract key medical values
        from the report text. This is much faster than using an LLM for extraction.
        """
        data = {}
        # Define a helper function to search for patterns
        def find_value(pattern, text_block):
            match = re.search(pattern, text_block, re.IGNORECASE | re.DOTALL)
            if match:
                # Return the first captured group, which should be the numeric value
                return float(match.group(1))
            return None

        # Define Regex patterns for each key metric
        patterns = {
            'hba1c_level': r"HBA1C.*?(\d{1,2}\.\d{1,2})",
            'bmi': r"BMI\s*:\s*(\d{1,2}\.\d{1,2})",
            'trestbps': r"Systolic BP.*?(\d{2,3})", # Systolic Blood Pressure
            'chol': r"TOTAL CHOLESTEROL.*?\n.*?(\d{2,3})",
            'fbs': r"GLUCOSE,\s*FASTING.*?\n.*?(\d{2,3})", # Fasting Blood Sugar
            'blood_glucose_level': r"GLUCOSE,\s*FASTING.*?\n.*?(\d{2,3})",
            'hemo': r"HAEMOGLOBIN\n.*?(\d{1,2}\.\d{1,2}|\d{1,2})",
            'pcv': r"PCV\n.*?(\d{1,2}\.\d{1,2})",
            'urea': r"UREA\n.*?(\d{1,3}\.\d{1,2})",
            'creatinine': r"CREATININE\n.*?(\d{1,2}\.\d{1,2})",
            'TSH': r"HORMONE \(TSH\)\n.*?(\d{1,2}\.\d{1,3})",
            'T3': r"TRI-IODOTHYRONINE \(13.*?(\d{1,2}\.\d{1,2})",
            'TT4': r"THYROXINE \(14.*?(\d{1,2}\.\d{1,2})"
        }

        for key, pattern in patterns.items():
            value = find_value(pattern, text)
            if value is not None:
                data[key] = value

        # Handle special case for 'fbs' (Fasting Blood Sugar > 120 is 1, else 0)
        if 'fbs' in data and data['fbs'] is not None:
            data['fbs'] = 1 if data['fbs'] > 120 else 0

        return data

    def process_file_from_bytes(self, file_bytes: bytes, filename: str) -> Tuple[str, str]:
        """Processes an uploaded file from a byte stream."""
        file_extension = os.path.splitext(filename)[1].lower()
        text = ""

        try:
            if file_extension == ".pdf":
                text = self._extract_text_from_pdf_bytes(file_bytes)
                file_type = "PDF Report"
            elif file_extension in [".png", ".jpg", ".jpeg"]:
                text = self._extract_text_from_image_bytes(file_bytes)
                file_type = "Image Report"
            elif file_extension == ".docx":
                text = self._extract_text_from_docx_bytes(file_bytes)
                file_type = "Word Document"
            elif file_extension == ".txt":
                text = file_bytes.decode('utf-8')
                file_type = "Text File"
            else:
                return "Unsupported file type", "Unsupported"
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return "", "Error"

        return text, file_type

    def _extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Extracts text from a PDF file provided as bytes."""
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    def _extract_text_from_image_bytes(self, image_bytes: bytes) -> str:
        """Extracts text from an image file provided as bytes using OCR."""
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text

    def _extract_text_from_docx_bytes(self, docx_bytes: bytes) -> str:
        """Extracts text from a DOCX file provided as bytes."""
        text = ""
        doc = Document(io.BytesIO(docx_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    def validate_report(self, text: str) -> Tuple[bool, str]:
        """A simple validation to check if the text seems like a medical report."""
        keywords = ['patient', 'report', 'doctor', 'clinic', 'lab', 'test', 'result']
        text_lower = text.lower()
        
        found_keywords = sum(1 for keyword in keywords if keyword in text_lower)
        
        if len(text) < 50:
            return False, "Report content is too short."
        if found_keywords < 2:
            return False, "The document does not appear to be a medical report."
            
        return True, "Report appears valid."