import pytest
import pandas as pd
from src.utils.validators import (
    validate_gene_id,
    validate_chem_id,
    validate_smiles,
    validate_protein_sequence,
    validate_batch_dataframe
)
from src.utils.exceptions import ValidationException

# --- Tests for validate_gene_id ---

def test_validate_gene_id_valid():
    assert validate_gene_id("CYP2D6") == "CYP2D6"
    assert validate_gene_id("  BRCA1 ") == "BRCA1"
    assert validate_gene_id("P12345") == "P12345"
    assert validate_gene_id("A0A0B4J2F0") == "A0A0B4J2F0"
    assert validate_gene_id("PA123") == "PA123"
    assert validate_gene_id("12345") == "12345"
    assert validate_gene_id("gene-with-hyphen.1") == "gene-with-hyphen.1"

def test_validate_gene_id_invalid_empty():
    with pytest.raises(ValidationException, match="cannot be empty"):
        validate_gene_id("")
    with pytest.raises(ValidationException, match="cannot be empty"):
        validate_gene_id("   ")

def test_validate_gene_id_invalid_type():
    with pytest.raises(ValidationException, match="must be a string"):
        validate_gene_id(None)
    with pytest.raises(ValidationException, match="must be a string"):
        validate_gene_id(123)

def test_validate_gene_id_invalid_short():
    with pytest.raises(ValidationException, match="too short"):
        validate_gene_id("A")

def test_validate_gene_id_invalid_long():
    with pytest.raises(ValidationException, match="too long"):
        validate_gene_id("A" * 51)

def test_validate_gene_id_invalid_chars():
    with pytest.raises(ValidationException, match="invalid characters"):
        validate_gene_id("GENE$")
    with pytest.raises(ValidationException, match="invalid characters"):
        validate_gene_id("GENE#1")
    with pytest.raises(ValidationException, match="invalid characters"):
        validate_gene_id("GENE 1") # Spaces not allowed

# --- Tests for validate_chem_id ---

def test_validate_chem_id_valid():
    assert validate_chem_id("Aspirin") == "Aspirin"
    assert validate_chem_id("  Ibuprofen (Advil) ") == "Ibuprofen (Advil)"
    assert validate_chem_id("12345") == "12345"
    assert validate_chem_id("PA161") == "PA161"
    assert validate_chem_id("CHEMBL123") == "CHEMBL123"
    assert validate_chem_id("drug-with-hyphen.2") == "drug-with-hyphen.2"
    assert validate_chem_id("Test[drug]") == "Test[drug]"

def test_validate_chem_id_invalid_empty():
    with pytest.raises(ValidationException, match="cannot be empty"):
        validate_chem_id("")

def test_validate_chem_id_invalid_short():
    with pytest.raises(ValidationException, match="too short"):
        validate_chem_id("A")

def test_validate_chem_id_invalid_long():
    with pytest.raises(ValidationException, match="too long"):
        validate_chem_id("A" * 101)

def test_validate_chem_id_invalid_chars():
    with pytest.raises(ValidationException, match="invalid characters"):
        validate_chem_id("Drug$")
    with pytest.raises(ValidationException, match="invalid characters"):
        validate_chem_id("Drug#@!")

# --- Tests for validate_smiles ---
# (Mocking RDKit is complex, so we test its known behavior)

def test_validate_smiles_valid():
    # Aspirin
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    canonical, is_valid = validate_smiles(smiles)
    assert is_valid
    assert canonical == "CC(=O)Oc1ccccc1C(=O)O" # RDKit canonicalizes

def test_validate_smiles_invalid_rdkit():
    # Invalid structure
    smiles = "C(C)C)C1=CC=C"
    canonical, is_valid = validate_smiles(smiles)
    assert not is_valid
    assert canonical == smiles # Returns original on failure

def test_validate_smiles_invalid_chars():
    smiles = "CC(=O)Oc1ccccc1C(=O)O_INVALID"
    canonical, is_valid = validate_smiles(smiles)
    assert not is_valid
    assert canonical == smiles

def test_validate_smiles_empty():
    canonical, is_valid = validate_smiles("")
    assert not is_valid
    assert canonical == ""

def test_validate_smiles_none():
    canonical, is_valid = validate_smiles(None)
    assert not is_valid
    assert canonical == ""

# --- Tests for validate_protein_sequence ---

def test_validate_protein_sequence_valid():
    seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY" # Valid, len 40
    cleaned, is_valid = validate_protein_sequence(seq)
    assert is_valid
    assert cleaned == seq

def test_validate_protein_sequence_valid_with_unknown():
    seq = "ACDEFGHIKLMNPQRSTVWYXXXX" # Valid, <10% 'X'
    cleaned, is_valid = validate_protein_sequence(seq)
    assert is_valid
    assert cleaned == seq

def test_validate_protein_sequence_valid_whitespace():
    seq = "  ACDEFGHIKLMNPQRSTVWY\nACDEFGHIKLMNPQRSTVWY  "
    cleaned, is_valid = validate_protein_sequence(seq)
    assert is_valid
    assert cleaned == "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

def test_validate_protein_sequence_invalid_short():
    seq = "ACDEFGH" # len 7
    cleaned, is_valid = validate_protein_sequence(seq)
    assert not is_valid
    assert cleaned == seq

def test_validate_protein_sequence_invalid_chars():
    seq = "ACDEFGHIKLMNPQRSTVWY_INVALID_BJZ"
    cleaned, is_valid = validate_protein_sequence(seq)
    assert not is_valid
    assert cleaned == seq

def test_validate_protein_sequence_invalid_too_many_unknown():
    seq = "XXXXXXXXXXXXXXXXXXXXYYYYYYYYYYYYYYYYYYYY" # 50% 'X'
    cleaned, is_valid = validate_protein_sequence(seq)
    assert not is_valid
    assert cleaned == seq

# --- Tests for validate_batch_dataframe ---

@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "gene_id": ["BRCA1", "CYP2D6", "EGFR"],
        "chem_id": ["PA123", "CHEMBL123", "Aspirin"]
    })

@pytest.fixture
def invalid_df_missing_cols():
    return pd.DataFrame({"gene_id": ["BRCA1"]})

@pytest.fixture
def invalid_df_nulls():
    return pd.DataFrame({
        "gene_id": ["BRCA1", None, "EGFR"],
        "chem_id": ["PA123", "CHEMBL123", None]
    })

@pytest.fixture
def invalid_df_bad_data():
    return pd.DataFrame({
        "gene_id": ["BRCA1", "BAD$GENE"],
        "chem_id": ["PA123", "BAD#CHEM"]
    })

def test_validate_batch_dataframe_valid(valid_df):
    errors = validate_batch_dataframe(valid_df)
    assert len(errors) == 0

def test_validate_batch_dataframe_invalid_type():
    errors = validate_batch_dataframe("not a dataframe")
    assert len(errors) == 1
    assert "not a pandas DataFrame" in errors[0]

def test_validate_batch_dataframe_empty():
    errors = validate_batch_dataframe(pd.DataFrame(columns=["gene_id", "chem_id"]))
    assert len(errors) == 1
    assert "DataFrame is empty" in errors[0]

def test_validate_batch_dataframe_missing_cols(invalid_df_missing_cols):
    errors = validate_batch_dataframe(invalid_df_missing_cols)
    assert len(errors) == 1
    assert "Missing required columns: chem_id" in errors[0]

def test_validate_batch_dataframe_too_large():
    df = pd.DataFrame({
        "gene_id": ["G"] * 1001,
        "chem_id": ["C"] * 1001
    })
    errors = validate_batch_dataframe(df)
    assert len(errors) > 0
    assert "Too many rows (1001)" in errors[0]

def test_validate_batch_dataframe_nulls(invalid_df_nulls):
    errors = validate_batch_dataframe(invalid_df_nulls)
    assert len(errors) == 2
    assert "null 'gene_id' values" in errors[0]
    assert "null 'chem_id' values" in errors[1]

def test_validate_batch_dataframe_bad_data(invalid_df_bad_data):
    errors = validate_batch_dataframe(invalid_df_bad_data)
    # 2 errors from sample check + 1 header
    assert len(errors) == 3
    assert "Row 1: Invalid gene_id ('BAD$GENE') - Gene ID contains invalid characters" in errors[1]
    assert "Row 1: Invalid chem_id ('BAD#CHEM') - Chemical ID contains invalid characters" in errors[2]
