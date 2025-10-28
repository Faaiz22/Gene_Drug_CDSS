"""
Input validation utilities for Drug-Gene CDSS.
"""

import re
from typing import Optional, List, Tuple
from .exceptions import ValidationException


def validate_gene_id(gene_id: str) -> str:
    """
    Validate gene identifier format.
    
    Accepts:
    - Gene symbols (e.g., CYP2D6, BRCA1)
    - UniProt accessions (e.g., P12345, A0A0B4J2F0)
    - PharmGKB IDs (e.g., PA123)
    - Entrez Gene IDs (numeric)
    
    Returns:
        Cleaned gene ID
    
    Raises:
        ValidationException: If format is invalid
    """
    if not gene_id or not isinstance(gene_id, str):
        raise ValidationException("Gene ID cannot be empty")
    
    gene_id = gene_id.strip()
    
    if len(gene_id) < 2:
        raise ValidationException(f"Gene ID too short: {gene_id}")
    
    if len(gene_id) > 50:
        raise ValidationException(f"Gene ID too long (max 50 chars): {gene_id}")
    
    # Check for invalid characters
    if re.search(r'[^\w\-\.]', gene_id):
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
        Cleaned chemical ID
    
    Raises:
        ValidationException: If format is invalid
    """
    if not chem_id or not isinstance(chem_id, str):
        raise ValidationException("Chemical ID cannot be empty")
    
    chem_id = chem_id.strip()
    
    if len(chem_id) < 2:
        raise ValidationException(f"Chemical ID too short: {chem_id}")
    
    if len(chem_id) > 100:
        raise ValidationException(f"Chemical ID too long (max 100 chars): {chem_id}")
    
    # Allow more characters for drug names (spaces, parens, etc.)
    if re.search(r'[^\w\s\-\.\(\)\[\]]', chem_id):
        raise ValidationException(
            f"Chemical ID contains invalid characters: {chem_id}"
        )
    
    return chem_id


def validate_smiles(smiles: str) -> Tuple[str, bool]:
    """
    Validate SMILES string.
    
    Returns:
        (cleaned_smiles, is_valid)
    """
    if not smiles or not isinstance(smiles, str):
        return "", False
    
    smiles = smiles.strip()
    
    if len(smiles) < 2 or len(smiles) > 1000:
        return smiles, False
    
    # Basic SMILES character check
    valid_chars = set('CNOPSFClBrI[]()=#-+\\/@0123456789cnops')
    if not all(c in valid_chars for c in smiles):
        return smiles, False
    
    # RDKit validation
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, False
        
        # Canonicalize
        canonical_smiles = Chem.MolToSmiles(mol)
        return canonical_smiles, True
    except:
        return smiles, False


def validate_protein_sequence(sequence: str) -> Tuple[str, bool]:
    """
    Validate protein amino acid sequence.
    
    Returns:
        (cleaned_sequence, is_valid)
    """
    if not sequence or not isinstance(sequence, str):
        return "", False
    
    # Remove whitespace and convert to uppercase
    sequence = ''.join(sequence.split()).upper()
    
    # Check length
    if len(sequence) < 20:
        return sequence, False
    
    if len(sequence) > 10000:
        return sequence, False
    
    # Valid amino acid codes
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Allow X (unknown) and U (selenocysteine) but warn
    extended_aa = valid_aa | {'X', 'U'}
    
    # Check if all characters are valid
    if not all(c in extended_aa for c in sequence):
        return sequence, False
    
    # Check for too many unknown residues
    unknown_count = sequence.count('X')
    if unknown_count / len(sequence) > 0.1:  # More than 10% unknown
        return sequence, False
    
    return sequence, True


def validate_batch_dataframe(df) -> List[str]:
    """
    Validate batch analysis dataframe.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    import pandas as pd
    
    errors = []
    
    if not isinstance(df, pd.DataFrame):
        errors.append("Input is not a pandas DataFrame")
        return errors
    
    # Check required columns
    required_cols = ['gene_id', 'chem_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        return errors
    
    # Check for empty dataframe
    if len(df) == 0:
        errors.append("DataFrame is empty")
        return errors
    
    # Check size limit
    if len(df) > 1000:
        errors.append(f"Too many rows ({len(df)}). Maximum 1000 allowed.")
    
    # Check for null values
    null_genes = df['gene_id'].isna().sum()
    null_chems = df['chem_id'].isna().sum()
    
    if null_genes > 0:
        errors.append(f"Found {null_genes} null gene_id values")
    
    if null_chems > 0:
        errors.append(f"Found {null_chems} null chem_id values")
    
    # Validate individual IDs (sample check)
    sample_size = min(10, len(df))
    sample = df.sample(n=sample_size)
    
    for idx, row in sample.iterrows():
        try:
            validate_gene_id(str(row['gene_id']))
        except ValidationException as e:
            errors.append(f"Row {idx}: Invalid gene_id - {e.message}")
        
        try:
            validate_chem_id(str(row['chem_id']))
        except ValidationException as e:
            errors.append(f"Row {idx}: Invalid chem_id - {e.message}")
    
    return errors
