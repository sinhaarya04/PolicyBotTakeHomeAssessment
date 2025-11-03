"""
DATA LOADER - Handles Loading CSV Files

This module is responsible for loading all the data we need:
- Medical policy documents (the text we want to analyze)
- HCPCS codes database (procedure codes like 99213, 70551)
- ICD-10 codes database (diagnosis codes like E11.9, I10)
- Ground truth labels (for testing accuracy)

WHAT IT DOES:
- Reads CSV files and converts them to Python dictionaries
- Handles different column name formats
- Provides error handling if files are missing or corrupted
- Stores all data in memory for fast access

WHY WE NEED THIS:
- Medical codes are stored in separate CSV files
- We need to load them once and reuse for all documents
- This makes the main processing much faster
"""

import pandas as pd         # For reading CSV files
import logging              # For showing progress messages
from typing import Dict, Tuple  # For type hints

logger = logging.getLogger(__name__)

class DataLoader:
    """
    DATA LOADER CLASS
    
    This class handles all the file loading operations.
    Think of it as a librarian that knows where all the books are.
    
    WHAT IT STORES:
    - hcpcs_codes: Dictionary of procedure codes (99213 -> "Office visit")
    - icd10_codes: Dictionary of diagnosis codes (E11.9 -> "Type 2 diabetes")
    - policies: The actual policy documents we want to analyze
    - labels: Ground truth for testing (what codes should be found)
    """
    
    def __init__(self):
        # Initialize empty storage for all our data
        self.hcpcs_codes = {}      # Will store procedure codes
        self.icd10_codes = {}      # Will store diagnosis codes
        self.policies = None      # Will store policy documents
        self.labels = None        # Will store ground truth labels
    
    def load_hcpcs_codes(self, filepath: str) -> Dict[str, str]:
        """
        LOAD HCPCS CODES FROM CSV FILE
        
        HCPCS codes are procedure codes like:
        - 99213: Office visit
        - 70551: MRI of brain
        - 70450: CT scan of head
        
        WHAT THIS DOES:
        1. Reads the CSV file
        2. Handles different column names (some files use 'code', others use 'Code')
        3. Creates a dictionary: code -> description
        4. Stores it in self.hcpcs_codes for later use
        
        INPUT: filepath - path to the HCPCS CSV file
        OUTPUT: Dictionary of codes and descriptions
        """
        try:
            # Read the CSV file
            # To read only first 1000 rows: df = pd.read_csv(filepath, nrows=1000)
            df = pd.read_csv(filepath)
            
            # Handle different column name formats
            # Some files use 'code', others use 'Code' - we check both
            code_col = 'code' if 'code' in df.columns else 'Code'
            desc_col = 'description' if 'description' in df.columns else 'Description'
            
            # Create dictionary: code -> description
            # Example: {'99213': 'Office visit', '70551': 'MRI of brain'}
            self.hcpcs_codes = dict(zip(df[code_col], df[desc_col]))
            
            # Tell user how many codes we loaded
            logger.info(f"Loaded {len(self.hcpcs_codes)} HCPCS codes")
            return self.hcpcs_codes
            
        except Exception as e:
            # If something goes wrong, log the error and return empty dictionary
            logger.error(f"Failed to load HCPCS codes: {e}")
            return {}
    
    def load_icd10_codes(self, filepath: str) -> Dict[str, str]:
        """Load ICD-10 codes from CSV file."""
        try:
            # Read the CSV file
            # To read only first 1000 rows: df = pd.read_csv(filepath, nrows=1000)
            df = pd.read_csv(filepath)
            # Handle different column name formats
            code_col = 'code' if 'code' in df.columns else 'Codes'
            desc_col = 'description' if 'description' in df.columns else 'Description'
            self.icd10_codes = dict(zip(df[code_col], df[desc_col]))
            logger.info(f"Loaded {len(self.icd10_codes)} ICD-10 codes")
            return self.icd10_codes
        except Exception as e:
            logger.error(f"Failed to load ICD-10 codes: {e}")
            return {}
    
    def load_policies(self, filepath: str) -> pd.DataFrame:
        """Load policy data from CSV file."""
        try:
            self.policies = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.policies)} policies")
            return self.policies
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            return pd.DataFrame()
    
    def load_labels(self, filepath: str) -> pd.DataFrame:
        """Load ground truth labels from CSV file."""
        try:
            self.labels = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.labels)} labels")
            return self.labels
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            return pd.DataFrame()
    
    def get_policy_text(self, policy_id: str) -> str:
        """Get policy text by ID."""
        if self.policies is None:
            return ""
        
        policy = self.policies[self.policies['id'] == policy_id]
        if len(policy) > 0:
            return policy.iloc[0]['text']
        return ""
