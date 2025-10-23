import os
import httpx  # <-- Import httpx
import asyncio  # <-- Import asyncio
import time
from pathlib import Path
from typing import Optional, Dict
from rdkit import Chem
import streamlit as st

# Define a shared client factory for efficiency
@st.cache_resource
def get_async_client() -> httpx.AsyncClient:
    """Creates a cached asynchronous HTTP client."""
    headers = {"User-Agent": "DrugGeneCDSS/1.0"}
    client = httpx.AsyncClient(headers=headers, timeout=10.0, follow_redirects=True)
    return client

class DataEnricher:
    """
    Fetches protein sequences and chemical SMILES from external APIs asynchronously.
    Caches results locally for efficiency.
    """

    def __init__(self, config: Dict):
        self.config = config
        cache_dir = Path(config['paths']['cache_dir'])
        self.smiles_cache = cache_dir / "smiles"
        self.seq_cache = cache_dir / "sequences"
        os.makedirs(self.smiles_cache, exist_ok=True)
        os.makedirs(self.seq_cache, exist_ok=True)
        
        # Get the cached client
        self.client = get_async_client()
        
        self.pharmgkb_api = "https://api.pharmgkb.org/v1/data"
        self.stats = {
            'smiles_cache_hits': 0, 'smiles_api_calls': 0, 'smiles_failures': 0,
            'seq_cache_hits': 0, 'seq_api_calls': 0, 'seq_failures': 0
        }
        self.last_error = ""

    async def _get_pharmgkb_data(self, entity_type: str, entity_id: str) -> Optional[dict]:
        try:
            url = f"{self.pharmgkb_api}/{entity_type}/{entity_id}"
            r = await self.client.get(url)  # <-- Use await and self.client
            await asyncio.sleep(0.2)  # <-- Use asyncio.sleep
            if r.status_code == 200:
                return r.json()
            self.last_error = f"PharmGKB API error {r.status_code} for {entity_type} {entity_id}"
        except Exception as e:
            self.last_error = f"PharmGKB query failed for {entity_id}: {e}"
        return None

    @st.cache_data(ttl="6h")
    async def fetch_smiles(self, chem_id: str) -> Optional[str]:
        chem_id = str(chem_id).strip()
        cache_path = self.smiles_cache / f"{chem_id}.txt"
        if cache_path.exists():
            self.stats['smiles_cache_hits'] += 1
            with open(cache_path, "r") as f:
                smiles = f.read().strip()
            return smiles if Chem.MolFromSmiles(smiles) else None

        self.stats['smiles_api_calls'] += 1
        smiles, search_term, search_type = None, None, 'cid'

        if chem_id.startswith('PA'):
            data = await self._get_pharmgkb_data('chemical', chem_id) # <-- await
            if data:
                for ref in data.get('data', {}).get('crossReferences', []):
                    if ref.get('resource') == 'PubChem Compound':
                        search_term = ref.get('resourceId')
                        break
                if not search_term:  # Fallback to name
                    search_term = data.get('data', {}).get('name')
                    search_type = 'name'
        else:
            search_term, search_type = chem_id, 'auto'

        if not search_term:
            self.stats['smiles_failures'] += 1
            return None

        try:
            if search_type in ['cid', 'auto']:
                r = await self.client.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{search_term}/property/CanonicalSMILES/TXT") # <-- await
                if r.status_code == 200:
                    smiles = r.text.strip()
            if not smiles and search_type in ['name', 'auto']:
                r = await self.client.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{search_term}/property/CanonicalSMILES/TXT") # <-- await
                if r.status_code == 200:
                    smiles = r.text.strip()
            await asyncio.sleep(0.2) # <-- asyncio.sleep
        except Exception as e:
            self.last_error = f"PubChem query failed: {e}"

        if smiles and Chem.MolFromSmiles(smiles):
            with open(cache_path, "w") as f:
                f.write(smiles)
            return smiles

        self.stats['smiles_failures'] += 1
        return None

    @st.cache_data(ttl="6h")
    async def fetch_sequence(self, gene_id: str) -> Optional[str]:
        gene_id = str(gene_id).strip()
        cache_path = self.seq_cache / f"{gene_id}.txt"
        if cache_path.exists():
            self.stats['seq_cache_hits'] += 1
            with open(cache_path, "r") as f:
                return f.read().strip()

        self.stats['seq_api_calls'] += 1
        seq, search_term, search_type = "", None, 'accession'

        if gene_id.startswith('PA'):
            data = await self._get_pharmgkb_data('gene', gene_id) # <-- await
            if data:
                for ref in data.get('data', {}).get('crossReferences', []):
                    if ref.get('resource') == 'UniProtKB':
                        search_term = ref.get('resourceId')
                        break
                if not search_term:  # Fallback to symbol
                    search_term = data.get('data', {}).get('symbol')
                    search_type = 'gene'
        else:
            search_term, search_type = gene_id, 'auto'

        if not search_term:
            self.stats['seq_failures'] += 1
            return None

        try:
            if search_type in ['accession', 'auto']:
                r = await self.client.get(f"https://www.uniprot.org/uniprot/{search_term}.fasta") # <-- await
                if r.status_code == 200:
                    seq = "".join(r.text.split('\n')[1:])
            if not seq and search_type in ['gene', 'auto']:
                r = await self.client.get(f"https://rest.uniprot.org/uniprotkb/search?query=gene:{search_term}&format=fasta&size=1") # <-- await
                if r.status_code == 200 and '>' in r.text:
                    seq = "".join(r.text.split('\n')[1:])
            await asyncio.sleep(0.2) # <-- asyncio.sleep
        except Exception as e:
            self.last_error = f"UniProt query failed: {e}"

        if seq and len(seq) > 20:
            with open(cache_path, "w") as f:
                f.write(seq)
            return seq

        self.stats['seq_failures'] += 1
        return None
