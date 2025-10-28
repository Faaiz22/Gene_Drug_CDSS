# src/utils/api_clients.py (COMPLETE REPLACEMENT)
import os
import httpx
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict
from rdkit import Chem
import hashlib
import json
# Add to imports
from .exceptions import DataFetchException, format_user_error
 
class DataEnricher:
    """
    Fetches protein sequences and chemical SMILES from external APIs asynchronously.
    Caches results locally for efficiency.
    """
# Update fetch_smiles method
async def fetch_smiles(self, chem_id: str) -> str:  # Changed: no Optional, raises instead
    """
    Fetch SMILES with proper error handling.
    
    Raises:
        DataFetchException: If SMILES cannot be retrieved
    """
    chem_id = str(chem_id).strip()
    
    if not chem_id:
        raise DataFetchException(
            format_user_error('smiles_not_found', drug_id=chem_id),
            "Empty or whitespace-only identifier provided"
        )
    
    # ... existing cache check code ...
    
    # ... existing API call code ...
    
    if not smiles:
        raise DataFetchException(
            format_user_error('smiles_not_found', drug_id=chem_id),
            f"Tried CID and name searches. Last error: {self.last_error}"
        )
    
    # Validate SMILES
    if not Chem.MolFromSmiles(smiles):
        raise DataFetchException(
            format_user_error('invalid_smiles', smiles=smiles),
            f"SMILES validation failed for {chem_id}"
        )
    
    # Cache and return
    # ...
    return smiles


async def fetch_sequence(self, gene_id: str) -> str:  # No Optional
    """
    Fetch sequence with proper error handling.
    
    Raises:
        DataFetchException: If sequence cannot be retrieved
    """
    gene_id = str(gene_id).strip()
    
    if not gene_id:
        raise DataFetchException(
            format_user_error('sequence_not_found', gene_id=gene_id),
            "Empty or whitespace-only identifier provided"
        )
    
    # ... existing code ...
    
    if not seq or len(seq) < 20:
        raise DataFetchException(
            format_user_error('sequence_not_found', gene_id=gene_id),
            f"Sequence too short or not found. Last error: {self.last_error}"
        )
    
    return seq


    def __init__(self, config: Dict):
        self.config = config
        cache_dir = Path(config['paths']['cache_dir'])
        self.smiles_cache = cache_dir / "smiles"
        self.seq_cache = cache_dir / "sequences"
        os.makedirs(self.smiles_cache, exist_ok=True)
        os.makedirs(self.seq_cache, exist_ok=True)
        
        # Create client per-instance, not cached globally
        headers = {"User-Agent": "DrugGeneCDSS/1.0"}
        self.client = httpx.AsyncClient(
            headers=headers, 
            timeout=30.0,  # Increased from 10s
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        
        self.pharmgkb_api = "https://api.pharmgkb.org/v1/data"
        self.stats = {
            'smiles_cache_hits': 0, 'smiles_api_calls': 0, 'smiles_failures': 0,
            'seq_cache_hits': 0, 'seq_api_calls': 0, 'seq_failures': 0
        }
        self.last_error = ""
        self._request_lock = asyncio.Lock()  # Rate limiting
        self._last_request_time = 0

    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def _rate_limit(self, delay: float = 0.34):
        """Enforce rate limiting (3 requests/second = 0.33s delay)"""
        async with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < delay:
                await asyncio.sleep(delay - time_since_last)
            self._last_request_time = time.time()

    async def _get_pharmgkb_data(self, entity_type: str, entity_id: str) -> Optional[dict]:
        """Fetch data from PharmGKB API with retries"""
        url = f"{self.pharmgkb_api}/{entity_type}/{entity_id}"
        
        for attempt in range(3):
            try:
                await self._rate_limit()
                r = await self.client.get(url)
                
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:  # Rate limited
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    self.last_error = f"PharmGKB API error {r.status_code} for {entity_type} {entity_id}"
                    break
                    
            except httpx.TimeoutException:
                self.last_error = f"Timeout querying PharmGKB for {entity_id}"
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                self.last_error = f"PharmGKB query failed for {entity_id}: {e}"
                break
        
        return None

    def _get_cache_key(self, identifier: str) -> str:
        """Generate deterministic cache key"""
        return hashlib.md5(identifier.encode()).hexdigest()

    async def fetch_smiles(self, chem_id: str) -> Optional[str]:
        """Fetch SMILES with proper async caching"""
        chem_id = str(chem_id).strip()
        cache_key = self._get_cache_key(chem_id)
        cache_path = self.smiles_cache / f"{cache_key}.json"
        
        # Check file cache (synchronous, fast)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    smiles = data.get("smiles")
                    if smiles and Chem.MolFromSmiles(smiles):
                        self.stats['smiles_cache_hits'] += 1
                        return smiles
            except (json.JSONDecodeError, KeyError):
                cache_path.unlink(missing_ok=True)

        self.stats['smiles_api_calls'] += 1
        smiles, search_term, search_type = None, None, 'cid'

        # PharmGKB ID handling
        if chem_id.startswith('PA'):
            data = await self._get_pharmgkb_data('chemical', chem_id)
            if data:
                for ref in data.get('data', {}).get('crossReferences', []):
                    if ref.get('resource') == 'PubChem Compound':
                        search_term = ref.get('resourceId')
                        break
                if not search_term:
                    search_term = data.get('data', {}).get('name')
                    search_type = 'name'
        else:
            search_term, search_type = chem_id, 'auto'

        if not search_term:
            self.stats['smiles_failures'] += 1
            return None

        # PubChem API calls with retries
        pubchem_urls = []
        if search_type in ['cid', 'auto']:
            pubchem_urls.append(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{search_term}/property/CanonicalSMILES/TXT"
            )
        if search_type in ['name', 'auto']:
            pubchem_urls.append(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{search_term}/property/CanonicalSMILES/TXT"
            )

        for url in pubchem_urls:
            for attempt in range(3):
                try:
                    await self._rate_limit()
                    r = await self.client.get(url)
                    
                    if r.status_code == 200:
                        smiles = r.text.strip()
                        if smiles and Chem.MolFromSmiles(smiles):
                            # Cache the result
                            with open(cache_path, "w") as f:
                                json.dump({
                                    "smiles": smiles,
                                    "original_id": chem_id,
                                    "timestamp": time.time()
                                }, f)
                            return smiles
                    elif r.status_code == 404:
                        break  # Not found, try next URL
                        
                except httpx.TimeoutException:
                    if attempt < 2:
                        await asyncio.sleep(1)
                        continue
                except Exception as e:
                    self.last_error = f"PubChem query failed: {e}"
                    break

        self.stats['smiles_failures'] += 1
        return None

    async def fetch_sequence(self, gene_id: str) -> Optional[str]:
        """Fetch protein sequence with proper async caching"""
        gene_id = str(gene_id).strip()
        cache_key = self._get_cache_key(gene_id)
        cache_path = self.seq_cache / f"{cache_key}.json"
        
        # Check file cache
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    seq = data.get("sequence")
                    if seq and len(seq) > 20:
                        self.stats['seq_cache_hits'] += 1
                        return seq
            except (json.JSONDecodeError, KeyError):
                cache_path.unlink(missing_ok=True)

        self.stats['seq_api_calls'] += 1
        seq, search_term, search_type = "", None, 'accession'

        # PharmGKB ID handling
        if gene_id.startswith('PA'):
            data = await self._get_pharmgkb_data('gene', gene_id)
            if data:
                for ref in data.get('data', {}).get('crossReferences', []):
                    if ref.get('resource') == 'UniProtKB':
                        search_term = ref.get('resourceId')
                        break
                if not search_term:
                    search_term = data.get('data', {}).get('symbol')
                    search_type = 'gene'
        else:
            search_term, search_type = gene_id, 'auto'

        if not search_term:
            self.stats['seq_failures'] += 1
            return None

        # UniProt API calls with retries
        uniprot_urls = []
        if search_type in ['accession', 'auto']:
            uniprot_urls.append(f"https://rest.uniprot.org/uniprotkb/{search_term}.fasta")
        if search_type in ['gene', 'auto']:
            uniprot_urls.append(
                f"https://rest.uniprot.org/uniprotkb/search?query=gene:{search_term}&format=fasta&size=1"
            )

        for url in uniprot_urls:
            for attempt in range(3):
                try:
                    await self._rate_limit()
                    r = await self.client.get(url)
                    
                    if r.status_code == 200 and '>' in r.text:
                        lines = r.text.split('\n')
                        seq = "".join(line for line in lines[1:] if not line.startswith('>'))
                        
                        if seq and len(seq) > 20:
                            # Cache the result
                            with open(cache_path, "w") as f:
                                json.dump({
                                    "sequence": seq,
                                    "original_id": gene_id,
                                    "timestamp": time.time()
                                }, f)
                            return seq
                    elif r.status_code == 404:
                        break  # Not found, try next URL
                        
                except httpx.TimeoutException:
                    if attempt < 2:
                        await asyncio.sleep(1)
                        continue
                except Exception as e:
                    self.last_error = f"UniProt query failed: {e}"
                    break

        self.stats['seq_failures'] += 1
        return None
