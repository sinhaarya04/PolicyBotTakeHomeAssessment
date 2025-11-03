"""
MEDICAL CODE INFERENCE PIPELINE WITH HUMAN REVIEW FLAGGING

Key Features:
- Extracting using ClinicalBERT for high accuracy
- Flagging documents with low ICD-10 confidence for human reviewer
- Providing better confidence scores
- Enabling human reviewers to step in and label codes

Review Logic:
- Flag ICD-10 codes with confidence < 0.6
- Flag when no ICD-10 codes found
- Flag if average confidence below moderate threshold
- Human reviewer can verify or label codes
"""
import pandas as pd
import re
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
import argparse
import sys
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to load the AI library (transformers)
# This library lets us use pre-trained AI models
try:
    from transformers import AutoTokenizer, AutoModel  # AI model components
    import torch                                        # AI framework
    TRANSFORMERS_AVAILABLE = True
    logger.info("AI library found - using smart ClinicalBERT model")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("AI library not found - will use simple rules instead")


class EfficientBERTEncoder:
    """
    AI TEXT ENCODER - The "Smart Brain" of the Pipeline
    
    This class uses ClinicalBERT (a medical AI model) to understand medical text.
    Think of it as a medical expert that can read and understand complex medical language.
    """
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        """Initialize the AI encoder."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.available = False
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_bert()
        else:
            logger.warning("AI library not available - using simple rules instead")
    
    def _initialize_bert(self):
        """Initialize ClinicalBERT with progress tracking."""
        try:
            logger.info("Loading ClinicalBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Use CPU for stability
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.available = True
            logger.info("ClinicalBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ClinicalBERT: {e}")
            self.available = False
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using ClinicalBERT."""
        if not self.available or not text or not str(text).strip():
            return np.zeros(768)  # Return zero vector as fallback
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"BERT encoding failed: {e}")
            return np.zeros(768)


class EfficientCodeInference:
    """Efficient medical code inference with ClinicalBERT."""
    
    def __init__(self, review_threshold: float = 0.6):
        """
        Initialize the inference engine.
        
        Args:
            review_threshold: Minimum average confidence to avoid human review (default: 0.6)
        """
        # Initialize BERT encoder
        self.bert_encoder = EfficientBERTEncoder()
        
        # Review threshold for flagging documents
        self.review_threshold = review_threshold
        
        # Load medical codes
        self.hcpcs_codes = self._load_hcpcs_codes()
        self.icd10_codes = self._load_icd10_codes()
        
        # Precompute embeddings for codes (with progress bar)
        self.hcpcs_embeddings = {}
        self.icd10_embeddings = {}
        
        if self.bert_encoder.available:
            self._precompute_embeddings()
        
        # Rule-based patterns
        self.hcpcs_patterns = [
            r'\b(?:CPT|HCPCS)\s*:?\s*([A-Z]\d{4})\b',
            r'\b(?:CPT|HCPCS)\s*:?\s*([A-Z]{2}\d{3})\b',
            r'\b(?:CPT|HCPCS)\s*:?\s*([A-Z]{3}\d{2})\b',
            r'\b(?:CPT|HCPCS)\s*:?\s*([A-Z]{4}\d)\b',
            r'\b(?:CPT|HCPCS)\s*:?\s*(G\d{4})\b',
            r'\b(?:CPT|HCPCS)\s*:?\s*(J\d{4})\b',
            r'\b(?:CPT|HCPCS)\s*:?\s*(Q\d{4})\b',
            r'\b(?:CPT|HCPCS)\s*:?\s*(S\d{4})\b',
            r'\b(?:CPT|HCPCS)\s*:?\s*(\d{5})\b',
        ]
        
        self.icd10_patterns = [
            r'\b(?:ICD-?10|ICD)\s*:?\s*([A-Z]\d{2}\.?\d*)\b',
            r'\b(?:ICD-?10|ICD)\s*:?\s*([A-Z]\d{2})\b',
            r'\b(?:diagnosis|dx)\s*:?\s*([A-Z]\d{2}\.?\d*)\b',
        ]
        
        # Medical procedure keywords
        self.procedure_keywords = {
            'mri': ['70551', '70552', '70553'],
            'magnetic resonance imaging': ['70551', '70552', '70553'],
            'ct scan': ['70450', '70460', '70470'],
            'computed tomography': ['70450', '70460', '70470'],
            'ultrasound': ['76700', '76705', '76770'],
            'x-ray': ['70010', '70015', '70030'],
            'biopsy': ['10021', '10022', '10040'],
            'surgery': ['10021', '10040', '10060'],
            'injection': ['96365', '96366', '96367'],
            'office visit': ['99213', '99214', '99215'],
            'consultation': ['99242', '99243', '99244'],
            'emergency': ['99281', '99282', '99283'],
        }
        
        self.diagnosis_keywords = {
            'diabetes': ['E11.9', 'E10.9'],
            'hypertension': ['I10'],
            'cancer': ['C78.00', 'C79.00'],
            'pneumonia': ['J18.9'],
            'heart disease': ['I25.9'],
            'stroke': ['I63.9'],
            'migraine': ['G43.9'],
            'depression': ['F32.9'],
            'anxiety': ['F41.9'],
            'asthma': ['J45.9'],
        }
    
    def _load_hcpcs_codes(self) -> Dict[str, str]:
        """Load HCPCS codes."""
        try:
            # To read only first 1000 rows: df = pd.read_csv("hcpcs.csv", nrows=1000)
            df = pd.read_csv("hcpcs.csv")
            if 'code' in df.columns and 'description' in df.columns:
                return dict(zip(df['code'], df['description']))
            else:
                return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        except Exception as e:
            logger.error(f"Error loading HCPCS codes: {e}")
            return {}
    
    def _load_icd10_codes(self) -> Dict[str, str]:
        """Load ICD-10 codes."""
        try:
            # To read only first 1000 rows: df = pd.read_csv("icd10cm.csv", nrows=1000)
            df = pd.read_csv("icd10cm.csv")
            if 'Codes' in df.columns and 'Description' in df.columns:
                return dict(zip(df['Codes'], df['Description']))
            else:
                return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        except Exception as e:
            logger.error(f"Error loading ICD-10 codes: {e}")
            return {}
    
    def _precompute_embeddings(self):
        """Precompute embeddings for medical codes with progress tracking."""
        logger.info("Precomputing embeddings for medical codes...")
        
        # HCPCS codes - use ALL available codes
        hcpcs_items = list(self.hcpcs_codes.items())
        for code, description in tqdm(hcpcs_items, desc="HCPCS embeddings"):
            if description and pd.notna(description):
                embedding = self.bert_encoder.encode_text(str(description))
                if embedding is not None:
                    self.hcpcs_embeddings[code] = embedding
        
        # ICD-10 codes - use ALL available codes
        icd10_items = list(self.icd10_codes.items())
        for code, description in tqdm(icd10_items, desc="ICD-10 embeddings"):
            if description and pd.notna(description):
                embedding = self.bert_encoder.encode_text(str(description))
                if embedding is not None:
                    self.icd10_embeddings[code] = embedding
        
        logger.info(f"Precomputed {len(self.hcpcs_embeddings)} HCPCS and {len(self.icd10_embeddings)} ICD-10 embeddings")
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity."""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except:
            return 0.0
    
    def _extract_with_rules(self, text: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Extract codes using rule-based approach."""
        hcpcs_codes = []
        icd10_codes = []
        
        if not text or pd.isna(text):
            return hcpcs_codes, icd10_codes
        
        text = str(text)
        
        # Extract HCPCS codes with dynamic confidence
        for pattern in self.hcpcs_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                code = match.upper()
                if self._is_valid_hcpcs_code(code):
                    confidence = self._calculate_confidence(code, text, 'explicit')
                    hcpcs_codes.append((code, confidence))
        
        # Extract ICD-10 codes with dynamic confidence
        for pattern in self.icd10_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                code = match.upper()
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
    
    def _extract_with_semantics(self, text: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Extract codes using semantic matching."""
        hcpcs_codes = []
        icd10_codes = []
        
        if not self.bert_encoder.available or not text or pd.isna(text):
            return hcpcs_codes, icd10_codes
        
        try:
            text_embedding = self.bert_encoder.encode_text(str(text))
            if text_embedding is None:
                return hcpcs_codes, icd10_codes
            
            # Find similar HCPCS codes with dynamic confidence
            for code, code_embedding in self.hcpcs_embeddings.items():
                similarity = self._compute_similarity(text_embedding, code_embedding)
                if similarity >= 0.3:  # Threshold
                    confidence = self._calculate_confidence(code, text, 'semantic', similarity)
                    hcpcs_codes.append((code, confidence))
            
            # Find similar ICD-10 codes with dynamic confidence
            for code, code_embedding in self.icd10_embeddings.items():
                similarity = self._compute_similarity(text_embedding, code_embedding)
                if similarity >= 0.3:  # Threshold
                    confidence = self._calculate_confidence(code, text, 'semantic', similarity)
                    icd10_codes.append((code, confidence))
            
        except Exception as e:
            logger.error(f"Semantic extraction failed: {e}")
        
        return hcpcs_codes, icd10_codes
    
    def _is_valid_hcpcs_code(self, code: str) -> bool:
        """Validate HCPCS code format."""
        if not code or len(code) < 3:
            return False
        return len(code) == 5 and (code[0].isalpha() or code.isdigit())
    
    def _is_valid_icd10_code(self, code: str) -> bool:
        """Validate ICD-10 code format."""
        if not code or len(code) < 3:
            return False
        pattern = r'^[A-Z]\d{2}(\.\d+)?$'
        return bool(re.match(pattern, code))
    
    def infer_codes(self, text: str) -> Tuple[List[str], List[str], List[float]]:
        """Infer codes using hybrid approach."""
        if not text or pd.isna(text) or not str(text).strip():
            return [], [], []
        
        text = str(text)
        
        # Extract with rules
        rule_hcpcs, rule_icd10 = self._extract_with_rules(text)
        
        # Extract with semantics (if available)
        semantic_hcpcs, semantic_icd10 = self._extract_with_semantics(text)
        
        # Combine results
        all_hcpcs = rule_hcpcs + semantic_hcpcs
        all_icd10 = rule_icd10 + semantic_icd10
        
        # Deduplicate and sort by confidence
        hcpcs_dict = {}
        for code, conf in all_hcpcs:
            if code not in hcpcs_dict or conf > hcpcs_dict[code]:
                hcpcs_dict[code] = conf
        
        icd10_dict = {}
        for code, conf in all_icd10:
            if code not in icd10_dict or conf > icd10_dict[code]:
                icd10_dict[code] = conf
        
        # Filter by confidence and limit results
        hcpcs_filtered = [(code, conf) for code, conf in hcpcs_dict.items() if conf >= 0.3]
        icd10_filtered = [(code, conf) for code, conf in icd10_dict.items() if conf >= 0.3]
        
        hcpcs_filtered.sort(key=lambda x: x[1], reverse=True)
        icd10_filtered.sort(key=lambda x: x[1], reverse=True)
        
        # Extract final results
        hcpcs_codes = [code for code, _ in hcpcs_filtered[:10]]
        icd10_codes = [code for code, _ in icd10_filtered[:10]]
        confidence_scores = [conf for _, conf in hcpcs_filtered + icd10_filtered]
        
        return hcpcs_codes, icd10_codes, confidence_scores
    
    def _should_flag_review(self, icd10_codes: List[str], icd10_confidences: List[float]) -> Tuple[bool, str]:
        """
        Determine if document should be flagged for human review.
        
        Review logic:
        - Flag when no ICD-10 codes found
        - Flag when average ICD-10 confidence < review_threshold (default 0.6)
        - Flag when all ICD-10 codes have confidence < 0.5
        
        Args:
            icd10_codes: List of ICD-10 codes found
            icd10_confidences: List of confidence scores for ICD-10 codes
            
        Returns:
            Tuple of (should_flag: bool, reason: str)
        """
        # Flag if no ICD-10 codes found
        if len(icd10_codes) == 0:
            return True, "No ICD-10 codes found - may need review"
        
        # Calculate average confidence
        avg_icd10_confidence = sum(icd10_confidences) / len(icd10_confidences) if icd10_confidences else 0.0
        
        # Flag if average confidence below threshold
        if avg_icd10_confidence < self.review_threshold:
            return True, f"Low average confidence ({avg_icd10_confidence:.2f} < {self.review_threshold})"
        
        # Flag if all codes have very low confidence
        if all(conf < 0.5 for conf in icd10_confidences):
            return True, f"All ICD-10 codes have confidence < 0.5"
        
        return False, ""
    
    def infer_codes_with_review(self, text: str) -> Dict:
        """
        Infer codes and determine if human review is needed.
        
        Args:
            text: Policy text to analyze
            
        Returns:
            Dictionary with:
            - hcpcs_codes: List of HCPCS codes
            - icd10_codes: List of ICD-10 codes
            - hcpcs_confidences: List of HCPCS confidence scores
            - icd10_confidences: List of ICD-10 confidence scores
            - needs_review: Boolean indicating if review is needed
            - review_reason: Reason for review flagging (if applicable)
        """
        if not text or pd.isna(text) or not str(text).strip():
            return {
                'hcpcs_codes': [],
                'icd10_codes': [],
                'hcpcs_confidences': [],
                'icd10_confidences': [],
                'needs_review': True,
                'review_reason': "Empty or invalid text"
            }
        
        # Extract codes with confidence scores from both methods
        rule_hcpcs, rule_icd10 = self._extract_with_rules(text)
        semantic_hcpcs, semantic_icd10 = self._extract_with_semantics(text)
        
        # Combine and deduplicate HCPCS codes with confidences
        hcpcs_dict = {}
        for code, conf in rule_hcpcs + semantic_hcpcs:
            if code not in hcpcs_dict or conf > hcpcs_dict[code]:
                hcpcs_dict[code] = conf
        
        # Combine and deduplicate ICD-10 codes with confidences
        icd10_dict = {}
        for code, conf in rule_icd10 + semantic_icd10:
            if code not in icd10_dict or conf > icd10_dict[code]:
                icd10_dict[code] = conf
        
        # Filter by confidence threshold and sort
        hcpcs_filtered = [(code, conf) for code, conf in hcpcs_dict.items() if conf >= 0.3]
        hcpcs_filtered.sort(key=lambda x: x[1], reverse=True)
        
        icd10_filtered = [(code, conf) for code, conf in icd10_dict.items() if conf >= 0.3]
        icd10_filtered.sort(key=lambda x: x[1], reverse=True)
        
        # Extract final results (top 10)
        hcpcs_codes = [code for code, _ in hcpcs_filtered[:10]]
        icd10_codes = [code for code, _ in icd10_filtered[:10]]
        hcpcs_confidences = [conf for _, conf in hcpcs_filtered[:10]]
        icd10_confidences = [conf for _, conf in icd10_filtered[:10]]
        
        # Check if review is needed
        needs_review, review_reason = self._should_flag_review(icd10_codes, icd10_confidences)
        
        return {
            'hcpcs_codes': hcpcs_codes,
            'icd10_codes': icd10_codes,
            'hcpcs_confidences': hcpcs_confidences,
            'icd10_confidences': icd10_confidences,
            'needs_review': needs_review,
            'review_reason': review_reason if needs_review else ""
        }


def process_policies_with_review(input_file: str, output_file: str, review_threshold: float = 0.6):
    """
    Process policies with human review flagging.
    
    Args:
        input_file: Input CSV file with policy texts
        output_file: Output CSV file for results
        review_threshold: Minimum average confidence to avoid review (default: 0.6)
    """
    logger.info(f"Loading policies from {input_file}...")
    
    # Load policies
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} policies")
    except Exception as e:
        logger.error(f"Error loading policies: {e}")
        return
    
    # Find text column
    text_column = None
    for col in ['cleaned_policy_text', 'policy_text', 'text']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        logger.error("No text column found in input file")
        return
    
    # Initialize inference engine
    logger.info("Initializing inference engine...")
    inference_engine = EfficientCodeInference(review_threshold=review_threshold)
    
    # Process each policy with progress bar
    results = []
    flagged_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing policies"):
        try:
            text = row[text_column]
            if pd.isna(text):
                text = ""
            
            policy_id = row.get('policy_id', idx)
            
            # Infer codes with review flagging
            result = inference_engine.infer_codes_with_review(text)
            
            if result['needs_review']:
                flagged_count += 1
            
            # Calculate average confidence
            all_confidences = result['hcpcs_confidences'] + result['icd10_confidences']
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            results.append({
                'id': policy_id,
                'hcpcs_codes': result['hcpcs_codes'],
                'icd10_codes': result['icd10_codes'],
                'hcpcs_confidences': result['hcpcs_confidences'],
                'icd10_confidences': result['icd10_confidences'],
                'average_confidence': avg_confidence,
                'needs_review': result['needs_review'],
                'review_reason': result['review_reason']
            })
                
        except Exception as e:
            logger.error(f"Error processing policy {idx}: {e}")
            results.append({
                'id': row.get('policy_id', idx),
                'hcpcs_codes': [],
                'icd10_codes': [],
                'hcpcs_confidences': [],
                'icd10_confidences': [],
                'average_confidence': 0.0,
                'needs_review': True,
                'review_reason': f"Error processing: {str(e)}"
            })
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Save results
    output_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    total_policies = len(output_df)
    policies_with_codes = len(output_df[output_df['icd10_codes'].apply(len) > 0])
    avg_confidence = output_df['average_confidence'].mean()
    
    logger.info(f"\nPipeline Summary:")
    logger.info(f"  Total policies processed: {total_policies}")
    logger.info(f"  Policies with ICD-10 codes: {policies_with_codes}")
    logger.info(f"  Policies flagged for review: {flagged_count} ({flagged_count/total_policies*100:.1f}%)")
    logger.info(f"  Average confidence: {avg_confidence:.3f}")
    
    return output_df


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="ClinicalBERT Medical Code Inference with Human Review Flagging")
    parser.add_argument("-input", required=True, help="Input CSV file with policy texts")
    parser.add_argument("-output", required=True, help="Output CSV file for inferred codes")
    parser.add_argument("-review-threshold", type=float, default=0.6, 
                       help="Minimum average ICD-10 confidence to avoid review (default: 0.6)")
    parser.add_argument("-verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} not found")
        sys.exit(1)
    
    try:
        # Process policies
        results_df = process_policies_with_review(
            args.input, 
            args.output, 
            review_threshold=args.review_threshold
        )
        
        if not results_df.empty:
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("No results generated")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()




    