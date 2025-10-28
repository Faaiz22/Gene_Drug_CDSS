"""
Input validation utilities for Drug-Gene CDSS.
"""

import re
from typing import Optional, List, Tuple
import pandas as pd
from rdkit import Chem
from .exceptions import ValidationException

# Compile regex patterns once for efficiency
RE_INVALID_GENE_CHARS = re.compile(r'[^\w\-\.]')
RE_INVALID_CHEM_CHARS = re.compile(r'[^\w\s\-\.\(\)\[\]]')

# Basic SMILES character check
# Note: This is a basic sanity check, not a full parser.
VALID_SMILES_CHARS = set('CNOPSFClBrI[]()=#-+\\/@0123456789cnops')

# Valid amino acid codes
VALID_AA_SET = set('ACDEFGHIKLMNPQRSTVWY')
# Allow X (unknown) and U (selenocysteine)
EXTENDED_AA_SET = VALID_AA_SET | {'X', 'U'}


def validate_gene_id(gene_id: str) -> str:
    """
    Validate gene identifier format.
    
    Accepts:
    - Gene symbols (e.g., CYP2D6, BRCA1)
    - UniProt accessions (e.g., P12345, A0A0B4J2F0)
    - PharmGKB IDs (e.g., PA123)
    - Entrez Gene IDs (numeric)
    
    Returns:
        Cleaned, stripped gene ID.
    
    Raises:
        ValidationException: If format is invalid.
    """
    if not isinstance(gene_id, str):
        raise ValidationException("Gene ID must be a string")
    
    gene_id = gene_id.strip()
    
    if not gene_id:
        raise ValidationException("Gene ID cannot be empty")
    
    if len(gene_id) < 2:
        raise ValidationException(f"Gene ID too short: {gene_id}")
    
    if len(gene_id) > 50:
        raise ValidationException(f"Gene ID too long (max 50 chars): {gene_id}")
    
    # Check for invalid characters
    if RE_INVALID_GENE_CHARS.search(gene_id):
        raise ValidationException(
            f"Gene ID contains invalid characters: {gene_id}. "
            "Only alphanumeric, hyphens, and periods allowed."
        )
    
    return gene_id


def validate_chem_id(chem_id: str) -> str:
    """
    Validate chemical identifier format.
    
    Accepts:
    - PubChem CIDs (numeric)
    - Drug names (alphanumeric with spaces/hyphens)
    - PharmGKB IDs (e.g., PA123)
    - ChEMBL IDs (e.g., CHEMBL123)
    
    Returns:
        Cleaned, stripped chemical ID.
    
    Raises:
        ValidationException: If format is invalid.
    """
    if not isinstance(chem_id, str):
        raise ValidationException("Chemical ID must be a string")
    
    chem_id = chem_id.strip()
    
    if not chem_id:
        raise ValidationException("Chemical ID cannot be empty")

    if len(chem_id) < 2:
        raise ValidationException(f"Chemical ID too short: {chem_id}")
    
    if len(chem_id) > 100:
        raise ValidationException(f"Chemical ID too long (max 100 chars): {chem_id}")
    
    # Allow more characters for drug names (spaces, parens, etc.)
    if RE_INVALID_CHEM_CHARS.search(chem_id):
        raise ValidationException(
            f"Chemical ID contains invalid characters: {chem_id}"
        )
    
    return chem_id


def validate_smiles(smiles: str) -> Tuple[str, bool]:
    """
    Validate and canonicalize a SMILES string using RDKit.
    
    Returns:
        (canonical_smiles, is_valid)
        If invalid, returns (original_smiles, False).
    """
    if not isinstance(smiles, str):
        return "", False
    
    smiles = smiles.strip()
    
    if len(smiles) < 2 or len(smiles) > 1000:
        return smiles, False
    
    # Basic SMILES character check
    if not all(c in VALID_SMILES_CHARS for c in smiles):
        return smiles, False
    
    # RDKit validation
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, False
        
        # Canonicalize
        canonical_smiles = Chem.MolToSmiles(mol)
        return canonical_smiles, True
    except Exception:
        # Catch any potential RDKit errors
        return smiles, False


def validate_protein_sequence(sequence: str) -> Tuple[str, bool]:
    """
    Validate protein amino acid sequence.
    
    Returns:
        (cleaned_sequence, is_valid)
    """
    if not isinstance(sequence, str):
        return "", False
    
    # Remove whitespace and convert to uppercase
    sequence = ''.join(sequence.split()).upper()
    
    # Check length
    if len(sequence) < 20: # Min length for most domains
        return sequence, False
    
    if len(sequence) > 10000: # Max length sanity check
        return sequence, False
    
    # Check if all characters are valid
    if not all(c in EXTENDED_AA_SET for c in sequence):
        return sequence, False
    
    # Check for too many unknown residues
    unknown_count = sequence.count('X')
    if len(sequence) > 0 and (unknown_count / len(sequence) > 0.1):  # More than 10% unknown
        return sequence, False
    
    return sequence, True


def validate_batch_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Validate batch analysis dataframe.
    
    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []
    
    if not isinstance(df, pd.DataFrame):
        errors.append("Input is not a pandas DataFrame")
        return errors
    
    # Check for empty dataframe
    if df.empty:
        errors.append("DataFrame is empty")
        return errors

    # Check required columns
    required_cols = ['gene_id', 'chem_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        return errors
    
    # Check size limit
    MAX_ROWS = 1000
    if len(df) > MAX_ROWS:
        errors.append(f"Too many rows ({len(df)}). Maximum {MAX_ROWS} allowed.")
        return errors # Stop further validation if too large
    
    # Check for null values
    null_genes = df['gene_id'].isna().sum()
    null_chems = df['chem_id'].isna().sum()
    
    if null_genes > 0:
        errors.append(f"Found {null_genes} null 'gene_id' values. Please fill or remove these rows.")
    
    if null_chems > 0:
        errors.append(f"Found {null_chems} null 'chem_id' values. Please fill or remove these rows.")
        
    if null_genes > 0 or null_chems > 0:
        return errors # Stop validation if nulls are present

    # Validate a sample of the data for performance
    sample_size = min(20, len(df))
    # Use head for deterministic sampling in tests, sample() is random
    sample = df.head(sample_size) 
    
    for idx, row in sample.iterrows():
        try:
            # Ensure data is string for validators
            validate_gene_id(str(row['gene_id']))
        except ValidationException as e:
            errors.append(f"Row {idx}: Invalid gene_id ('{row['gene_id']}') - {e.message}")
        
        try:
            validate_chem_id(str(row['chem_id']))
        except ValidationException as e:
            errors.append(f"Row {idx}: Invalid chem_id ('{row['chem_id']}') - {e.message}")
            
    if errors:
         errors.insert(0, f"Found {len(errors)} issues in the first {sample_size} rows:")

    return errors
