import pytest
import pytest_asyncio
import httpx
import json
import time
from pathlib import Path
from src.utils.api_clients import DataEnricher
from src.utils.exceptions import DataFetchException

# We need to mark all test functions as async
pytestmark = pytest.mark.asyncio

# Mock config for the enricher
@pytest.fixture
def mock_config(tmp_path):
    cache_dir = tmp_path / "test_cache"
    return {
        "paths": {"cache_dir": str(cache_dir)},
        "api_keys": {} # Not needed for these mocks
    }

# Mock RDKit's Chem.MolFromSmiles to always return True (valid)
@pytest.fixture(autouse=True)
def mock_rdkit(mocker):
    mock_mol = mocker.MagicMock()
    mocker.patch("src.utils.api_clients.Chem.MolFromSmiles", return_value=mock_mol)

# Async fixture to create and properly close the enricher client
@pytest_asyncio.fixture
async def enricher(mock_config):
    async with DataEnricher(mock_config) as enricher_instance:
        yield enricher_instance

# Fixture to mock the httpx.AsyncClient
@pytest.fixture
def mock_http_client(mocker):
    return mocker.MagicMock(spec=httpx.AsyncClient)

# Fixture to simulate a successful PubChem response
@pytest.fixture
def mock_pubchem_response_success(mocker):
    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.text = "CC(=O)Oc1ccccc1C(=O)O" # Aspirin SMILES
    return mock_response
    
# Fixture to simulate a successful UniProt response
@pytest.fixture
def mock_uniprot_response_success(mocker):
    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.text = ">sp|P0DP23|... \nACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY" # Valid seq
    return mock_response

# --- Tests for fetch_smiles ---

async def test_fetch_smiles_success_cid(enricher, mocker, mock_pubchem_response_success):
    # Mock the client 'get' method
    enricher.client = mocker.MagicMock(spec=httpx.AsyncClient)
    enricher.client.get = mocker.AsyncMock(return_value=mock_pubchem_response_success)
    
    smiles = await enricher.fetch_smiles("1234") # 1234 is numeric, will be treated as CID
    
    assert smiles == "CC(=O)Oc1ccccc1C(=O)O"
    enricher.client.get.assert_called_once_with(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/1234/property/CanonicalSMILES/TXT"
    )
    assert enricher.stats["smiles_api_calls"] == 1

async def test_fetch_smiles_success_name(enricher, mocker, mock_pubchem_response_success):
    mock_404 = mocker.MagicMock(spec=httpx.Response, status_code=404)
    
    enricher.client = mocker.MagicMock(spec=httpx.AsyncClient)
    enricher.client.get = mocker.AsyncMock(
        side_effect=[mock_404, mock_pubchem_response_success] # Fail CID lookup, succeed Name lookup
    )
    
    smiles = await enricher.fetch_smiles("Aspirin") # Not numeric, will try 'auto'
    
    assert smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert enricher.client.get.call_count == 2
    enricher.client.get.assert_any_call(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/Aspirin/property/CanonicalSMILES/TXT"
    )
    enricher.client.get.assert_called_with( # Last call
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/Aspirin/property/CanonicalSMILES/TXT"
    )
    assert enricher.stats["smiles_api_calls"] == 1

async def test_fetch_smiles_not_found(enricher, mocker):
    mock_404 = mocker.MagicMock(spec=httpx.Response, status_code=404)
    
    enricher.client = mocker.MagicMock(spec=httpx.AsyncClient)
    enricher.client.get = mocker.AsyncMock(return_value=mock_404) # Both CID and Name fail
    
    with pytest.raises(DataFetchException, match="Could not find molecular structure"):
        await enricher.fetch_smiles("FakeDrug123")
        
    assert enricher.stats["smiles_failures"] == 1
    assert enricher.client.get.call_count == 2 # Tries both URLs

async def test_fetch_smiles_api_error(enricher, mocker):
    mock_500 = mocker.MagicMock(spec=httpx.Response, status_code=500)
    
    enricher.client = mocker.MagicMock(spec=httpx.AsyncClient)
    # Simulate 3 retries failing for both URLs
    enricher.client.get = mocker.AsyncMock(return_value=mock_500) 
    
    with pytest.raises(DataFetchException, match="Could not find molecular structure"):
        await enricher.fetch_smiles("1234")
        
    assert enricher.stats["smiles_failures"] == 1
    assert enricher.client.get.call_count == 3 # 3 retries on first URL

async def test_fetch_smiles_empty_id(enricher):
    with pytest.raises(DataFetchException, match="Could not find molecular structure"):
        await enricher.fetch_smiles("  ")
    with pytest.raises(DataFetchException, match="Could not find molecular structure"):
        await enricher.fetch_smiles(None)

async def test_fetch_smiles_caching(enricher, mocker, mock_pubchem_response_success):
    enricher.client = mocker.MagicMock(spec=httpx.AsyncClient)
    enricher.client.get = mocker.AsyncMock(return_value=mock_pubchem_response_success)
    
    # First call - API
    smiles1 = await enricher.fetch_smiles("1234")
    assert smiles1 == "CC(=O)Oc1ccccc1C(=O)O"
    assert enricher.stats["smiles_api_calls"] == 1
    assert enricher.stats["smiles_cache_hits"] == 0
    enricher.client.get.assert_called_once()
    
    # Second call - Cache
    smiles2 = await enricher.fetch_smiles("1234")
    assert smiles2 == "CC(=O)Oc1ccccc1C(=O)O"
    assert enricher.stats["smiles_api_calls"] == 1 # No change
    assert enricher.stats["smiles_cache_hits"] == 1 # Incremented
    enricher.client.get.assert_called_once() # No new call

# --- Tests for fetch_sequence ---

async def test_fetch_sequence_success_accession(enricher, mocker, mock_uniprot_response_success):
    enricher.client = mocker.MagicMock(spec=httpx.AsyncClient)
    enricher.client.get = mocker.AsyncMock(return_value=mock_uniprot_response_success)
    
    seq = await enricher.fetch_sequence("P12345") # Will be treated as accession
    
    assert seq == "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    enricher.client.get.assert_called_once_with(
        "https://rest.uniprot.org/uniprotkb/P12345.fasta"
    )
    assert enricher.stats["seq_api_calls"] == 1

async def test_fetch_sequence_success_gene_name(enricher, mocker, mock_uniprot_response_success):
    mock_404 = mocker.MagicMock(spec=httpx.Response, status_code=404)
    
    enricher.client = mocker.MagicMock(spec=httpx.AsyncClient)
    enricher.client.get = mocker.AsyncMock(
        side_effect=[mock_404, mock_uniprot_response_success] # Fail accession, succeed gene query
    )
    
    seq = await enricher.fetch_sequence("BRCA1") # Will try accession first
    
    assert seq == "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    assert enricher.client.get.call_count == 2
    enricher.client.get.assert_called_with( # Last call
        "https://rest.uniprot.org/uniprotkb/search?query=gene:BRCA1&format=fasta&size=1"
    )
    assert enricher.stats["seq_api_calls"] == 1
    
async def test_fetch_sequence_not_found(enricher, mocker):
    mock_404 = mocker.MagicMock(spec=httpx.Response, status_code=404)
    
    enricher.client = mocker.MagicMock(spec=httpx.AsyncClient)
    enricher.client.get = mocker.AsyncMock(return_value=mock_404) # Both fail
    
    with pytest.raises(DataFetchException, match="Could not find protein sequence"):
        await enricher.fetch_sequence("FakeGene123")
        
    assert enricher.stats["seq_failures"] == 1
    assert enricher.client.get.call_count == 2 # Tries both URLs

async def test_fetch_sequence_caching(enricher, mocker, mock_uniprot_response_success):
    enricher.client = mocker.MagicMock(spec=httpx.AsyncClient)
    enricher.client.get = mocker.AsyncMock(return_value=mock_uniprot_response_success)
    
    # First call - API
    seq1 = await enricher.fetch_sequence("P12345")
    assert enricher.stats["seq_api_calls"] == 1
    assert enricher.stats["seq_cache_hits"] == 0
    
    # Second call - Cache
    seq2 = await enricher.fetch_sequence("P12345")
    assert seq1 == seq2
    assert enricher.stats["seq_api_calls"] == 1 # No change
    assert enricher.stats["seq_cache_hits"] == 1 # Incremented
    enricher.client.get.assert_called_once() # Still only one call
