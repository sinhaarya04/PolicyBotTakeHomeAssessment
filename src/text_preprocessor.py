"""
TEXT PREPROCESSOR - Cleans and Normalizes Text

This module is responsible for cleaning up the medical policy text
before we try to find codes in it.

WHAT IT DOES:
- Removes extra spaces and special characters
- Converts text to lowercase for consistent matching
- Expands medical abbreviations (MRI -> magnetic resonance imaging)
- Handles empty or missing text gracefully

WHY WE NEED THIS:
- Medical text can be messy with extra spaces, typos, etc.
- We need consistent formatting to match against our code database
- Some terms have multiple ways of being written (MRI vs magnetic resonance imaging)
"""

import re                   # For finding and replacing text patterns
import pandas as pd         # For handling data
import logging              # For showing progress messages
from typing import List     # For type hints

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    TEXT PREPROCESSOR CLASS
    
    This class cleans up medical policy text to make it easier to analyze.
    Think of it as a proofreader that standardizes the text.
    
    WHAT IT KNOWS:
    - Medical abbreviations and their full forms
    - How to clean up messy text
    - How to handle empty or missing text
    """
    
    def __init__(self):
        # Dictionary of medical abbreviations and their full forms
        # This helps us find codes even when abbreviations are used
        self.medical_terms = {
            'mri': 'magnetic resonance imaging',           # MRI -> magnetic resonance imaging
            'ct': 'computed tomography',                   # CT -> computed tomography
            'ultrasound': 'ultrasonography',              # Ultrasound -> ultrasonography
            'biopsy': 'tissue sampling',                   # Biopsy -> tissue sampling
            'surgery': 'surgical procedure'                # Surgery -> surgical procedure
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Basic normalization
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def extract_medical_terms(self, text: str) -> list:
        """Extract medical terms from text."""
        terms = []
        text_lower = text.lower()
        
        for term, expansion in self.medical_terms.items():
            if term in text_lower:
                terms.append(term)
                terms.append(expansion)
        
        return terms
    
    def clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove special characters but keep medical codes
        text = re.sub(r'[^\w\s\.\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
