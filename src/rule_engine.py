"""
RULE ENGINE - Finds Medical Codes Using Simple Rules

This is the "brain" of the fast pipeline. It uses simple rules and patterns
to find medical codes in policy documents.

WHAT IT DOES:
- Looks for codes that are already mentioned (like "CPT 99213")
- Finds keywords like "MRI" and suggests MRI codes
- Matches medical terms to appropriate codes
- Calculates confidence scores based on how specific the match is

HOW IT WORKS:
1. PATTERN MATCHING: Looks for codes in specific formats
2. KEYWORD MATCHING: Finds medical terms and suggests codes
3. CONFIDENCE CALCULATION: Scores how confident we are in each match

WHY WE NEED THIS:
- Not all codes are explicitly mentioned
- Some procedures are described in plain English
- We need to bridge the gap between medical language and billing codes
"""

import re                   # For finding patterns in text
import pandas as pd         # For handling data
import logging              # For showing progress messages
from typing import List, Tuple, Dict  # For type hints

logger = logging.getLogger(__name__)

class RuleEngine:
    """
    RULE ENGINE CLASS
    
    This is the main logic for finding medical codes using simple rules.
    Think of it as a smart assistant that knows medical terminology.
    
    WHAT IT KNOWS:
    - How to find codes that are already mentioned
    - Medical keywords and what codes they relate to
    - How to calculate confidence scores
    - Different formats for medical codes
    """
    
    def __init__(self):
        # PATTERN MATCHING - These are the formats we look for when codes are explicitly mentioned
        
        # HCPCS/CPT code patterns (procedure codes)
        self.hcpcs_patterns = [
            r'\b([A-Z]\d{4})\b',              # Standard HCPCS format: G0008, A9601
            r'\b(CPT\s*(\d{5}))\b',           # CPT format: CPT 99213, CPT99213
            r'\b(HCPCS\s*([A-Z]\d{4}))\b'     # HCPCS format: HCPCS G0008
        ]
        
        # ICD-10 code patterns (diagnosis codes)
        self.icd10_patterns = [
            r'\b([A-Z]\d{2}(?:\.\d{1,3})?)\b',                    # ICD-10 format: E11.9, I10
            r'\b(ICD-10\s*([A-Z]\d{2}(?:\.\d{1,3})?))\b'         # ICD-10 with prefix: ICD-10 E11.9
        ]
        
        # KEYWORD MATCHING - Medical terms and the codes they relate to
        
        # Medical procedure keywords (what procedures suggest what codes)
        self.procedure_keywords = {
            'mri': ['70551', '70552', '70553'],                    # MRI -> MRI codes
            'ct scan': ['70450', '70460', '70470'],                # CT scan -> CT codes
            'ultrasound': ['76700', '76705', '76706'],             # Ultrasound -> Ultrasound codes
            'biopsy': ['88305', '88307', '88309'],                 # Biopsy -> Biopsy codes
            'surgery': ['10021', '10040', '10060']                 # Surgery -> Surgery codes
        }
        
        # Medical diagnosis keywords (what conditions suggest what codes)
        self.diagnosis_keywords = {
            'diabetes': ['E11.9', 'E10.9'],                        # Diabetes -> Diabetes codes
            'hypertension': ['I10', 'I11.9'],                      # High blood pressure -> Hypertension codes
            'cancer': ['C78.00', 'C79.9'],                         # Cancer -> Cancer codes
            'pneumonia': ['J18.9', 'J15.9']                        # Pneumonia -> Pneumonia codes
        }
    
    def _is_valid_hcpcs_code(self, code: str) -> bool:
        """Validate HCPCS code format."""
        return bool(re.match(r'^[A-Z]\d{4}$', code))
    
    def _is_valid_icd10_code(self, code: str) -> bool:
        """Validate ICD-10 code format."""
        return bool(re.match(r'^[A-Z]\d{2}(?:\.\d{1,3})?$', code))
    
    def _calculate_confidence(self, code: str, text: str, match_type: str, match_strength: float = 1.0) -> float:
        """Calculate dynamic confidence score based on match quality."""
        base_confidence = {
            'explicit': 0.9,
            'keyword': 0.6,
            'semantic': 0.5
        }
        
        # Start with base confidence for match type
        confidence = base_confidence.get(match_type, 0.5)
        
        # Adjust based on match strength (for semantic matches)
        if match_type == 'semantic':
            confidence = match_strength * 0.8  # Scale to 0-0.8 range
        elif match_type == 'keyword':
            # Adjust based on keyword specificity
            keyword_specificity = self._get_keyword_specificity(text, code)
            confidence = base_confidence['keyword'] * keyword_specificity
        
        # Boost confidence for explicit mentions
        if match_type == 'explicit':
            # Check if code is mentioned in context
            context_boost = self._get_context_boost(text, code)
            confidence = min(0.95, confidence + context_boost)
        
        # Ensure confidence is within reasonable bounds
        return max(0.1, min(0.95, confidence))
    
    def _get_keyword_specificity(self, text: str, code: str) -> float:
        """Calculate keyword specificity score."""
        text_lower = text.lower()
        
        # More specific keywords get higher confidence
        specific_keywords = ['mri', 'ct scan', 'ultrasound', 'biopsy', 'surgery']
        general_keywords = ['test', 'exam', 'procedure', 'scan']
        
        specificity = 1.0
        for keyword in specific_keywords:
            if keyword in text_lower:
                specificity = min(1.2, specificity + 0.1)
        
        for keyword in general_keywords:
            if keyword in text_lower:
                specificity = max(0.8, specificity - 0.1)
        
        return specificity
    
    def _get_context_boost(self, text: str, code: str) -> float:
        """Calculate context boost for explicit mentions."""
        # Look for supporting context around the code
        context_indicators = ['required', 'covered', 'approved', 'eligible', 'indicated']
        text_lower = text.lower()
        
        boost = 0.0
        for indicator in context_indicators:
            if indicator in text_lower:
                boost += 0.05
        
        return min(0.1, boost)
    
    def extract_codes(self, text: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Extract codes using rule-based approach with dynamic confidence."""
        hcpcs_codes = []
        icd10_codes = []
        
        if not text or pd.isna(text):
            return hcpcs_codes, icd10_codes
        
        text = str(text)
        
        # Extract HCPCS codes with dynamic confidence
        for pattern in self.hcpcs_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                code = match.upper() if isinstance(match, str) else match[1].upper()
                if self._is_valid_hcpcs_code(code):
                    confidence = self._calculate_confidence(code, text, 'explicit')
                    hcpcs_codes.append((code, confidence))
        
        # Extract ICD-10 codes with dynamic confidence
        for pattern in self.icd10_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                code = match.upper() if isinstance(match, str) else match[1].upper()
                if self._is_valid_icd10_code(code):
                    confidence = self._calculate_confidence(code, text, 'explicit')
                    icd10_codes.append((code, confidence))
        
        # Keyword-based extraction with dynamic confidence
        text_lower = text.lower()
        for keyword, codes in self.procedure_keywords.items():
            if keyword in text_lower:
                for code in codes:
                    confidence = self._calculate_confidence(code, text, 'keyword')
                    hcpcs_codes.append((code, confidence))
        
        for keyword, codes in self.diagnosis_keywords.items():
            if keyword in text_lower:
                for code in codes:
                    confidence = self._calculate_confidence(code, text, 'keyword')
                    icd10_codes.append((code, confidence))
        
        return hcpcs_codes, icd10_codes
