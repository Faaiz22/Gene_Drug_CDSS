import os
import requests
import time
from pathlib import Path
from typing import Optional, Dict
from rdkit import Chem


import streamlit # <-- Import Streamlit

class DataEnricher:
    # ... (init method remains the same) ...

    def _get_pharmgkb_data(self, entity_type: str, entity_id: str) -> Optional[dict]:
        # ... (this method remains the same, it's called by the cached functions) ...
        try:
            url = f"{self.pharmgkb_api}/{entity_type}/{entity_id}"
            r = self.session.get(url)
            time.sleep(0.2)  # Rate limiting
            if r.status_code == 200: return r.json()
            self.last_error = f"PharmGKB API error {r.status_code} for {entity_type} {entity_id}"
        except Exception as e:
            self.last_error = f"PharmGKB query failed for {entity_id}: {e}"
        return None

    @st.cache_data(ttl="6h")  # <-- Cache results for 6 hours
    def fetch_smiles(self, chem_id: str) -> Optional[str]:
        _self = self # Use _self inside cached method
        chem_id = str(chem_id).strip()
        cache_path = _self.smiles_cache / f"{chem_id}.txt"
        if cache_path.exists():
            _self.stats['smiles_cache_hits'] += 1
            with open(cache_path, "r") as f: smiles = f.read().strip()
            return smiles if Chem.MolFromSmiles(smiles) else None

        _self.stats['smiles_api_calls'] += 1
        smiles, search_term, search_type = None, None, 'cid'

        if chem_id.startswith('PA'):
            data = _self._get_pharmgkb_data('chemical', chem_id)
            # ... (rest of the logic is the same, just use _self instead of self) ...
            if data:
                for ref in data.get('data', {}).get('crossReferences', []):
                    if ref.get('resource') == 'PubChem Compound':
                        search_term = ref.get('resourceId')
                        break
                if not search_term: # Fallback to name
                    search_term = data.get('data', {}).get('name')
                    search_type = 'name'
        else: search_term, search_type = chem_id, 'auto'

        if not search_term:
            _self.stats['smiles_failures'] += 1; return None
        
        # ... (rest of PubChem query logic is the same, use _self.session) ...
        try:
            if search_type in ['cid', 'auto']:
                r = _self.session.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{search_term}/property/CanonicalSMILES/TXT")
                if r.status_code == 200: smiles = r.text.strip()
            if not smiles and search_type in ['name', 'auto']:
                r = _self.session.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{search_term}/property/CanonicalSMILES/TXT")
                if r.status_code == 200: smiles = r.text.strip()
            time.sleep(0.2)
        except Exception as e: _self.last_error = f"PubChem query failed: {e}"


        if smiles and Chem.MolFromSmiles(smiles):
            with open(cache_path, "w") as f: f.write(smiles)
            return smiles

        _self.stats['smiles_failures'] += 1; return None

    @st.cache_data(ttl="6h")  # <-- Cache results for 6 hours
    def fetch_sequence(self, gene_id: str) -> Optional[str]:
        _self = self # Use _self inside cached method
        gene_id = str(gene_id).strip()
        cache_path = _self.seq_cache / f"{gene_id}.txt"
        if cache_path.exists():
            _self.stats['seq_cache_hits'] += 1
            with open(cache_path, "r") as f: return f.read().strip()

        _self.stats['seq_api_calls'] += 1
        seq, search_term, search_type = "", None, 'accession'

        if gene_id.startswith('PA'):
            data = _self._get_pharmgkb_data('gene', gene_id)
             # ... (rest of the logic is the same, just use _self instead of self) ...
            if data:
                for ref in data.get('data', {}).get('crossReferences', []):
                    if ref.get('resource') == 'UniProtKB':
                        search_term = ref.get('resourceId')
                        break
                if not search_term: # Fallback to symbol
                    search_term = data.get('data', {}).get('symbol')
                    search_type = 'gene'
        else: search_term, search_type = gene_id, 'auto'

        if not search_term:
            _self.stats['seq_failures'] += 1; return None
        
        # ... (rest of UniProt query logic is the same, use _self.session) ...
        try:
            if search_type in ['accession', 'auto']:
                r = _self.session.get(f"https://www.uniprot.org/uniprot/{search_term}.fasta")
                if r.status_code == 200: seq = "".join(r.text.split('\n')[1:])
            if not seq and search_type in ['gene', 'auto']:
                r = _self.session.get(f"https://rest.uniprot.org/uniprotkb/search?query=gene:{search_term}&format=fasta&size=1")
                if r.status_code == 200 and '>' in r.text: seq = "".join(r.text.split('\n')[1:])
            time.sleep(0.2)
        except Exception as e: _self.last_error = f"UniProt query failed: {e}"


        if seq and len(seq) > 20:
            with open(cache_path, "w") as f: f.write(seq)
            return seq

        _self.stats['seq_failures'] += 1; return None

class DataEnricher:
    """
    Fetches protein sequences and chemical SMILES from external APIs.
    Caches results locally for efficiency.
    Includes fallback logic: if direct ID mapping fails, it searches by name/symbol.
    """

    def __init__(self, config: Dict):
        self.config = config
        cache_dir = Path(config['paths']['cache_dir'])
        self.smiles_cache = cache_dir / "smiles"
        self.seq_cache = cache_dir / "sequences"
        os.makedirs(self.smiles_cache, exist_ok=True)
        os.makedirs(self.seq_cache, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "DrugGeneCDSS/1.0"})
        self.pharmgkb_api = "https://api.pharmgkb.org/v1/data"
        self.stats = {
            'smiles_cache_hits': 0, 'smiles_api_calls': 0, 'smiles_failures': 0,
            'seq_cache_hits': 0, 'seq_api_calls': 0, 'seq_failures': 0
        }
        self.last_error = ""

    def _get_pharmgkb_data(self, entity_type: str, entity_id: str) -> Optional[dict]:
        try:
            url = f"{self.pharmgkb_api}/{entity_type}/{entity_id}"
            r = self.session.get(url)
            time.sleep(0.2)  # Rate limiting
            if r.status_code == 200: return r.json()
            self.last_error = f"PharmGKB API error {r.status_code} for {entity_type} {entity_id}"
        except Exception as e:
            self.last_error = f"PharmGKB query failed for {entity_id}: {e}"
        return None

    def fetch_smiles(self, chem_id: str) -> Optional[str]:
        chem_id = str(chem_id).strip()
        cache_path = self.smiles_cache / f"{chem_id}.txt"
        if cache_path.exists():
            self.stats['smiles_cache_hits'] += 1
            with open(cache_path, "r") as f: smiles = f.read().strip()
            return smiles if Chem.MolFromSmiles(smiles) else None

        self.stats['smiles_api_calls'] += 1
        smiles, search_term, search_type = None, None, 'cid'

        if chem_id.startswith('PA'):
            data = self._get_pharmgkb_data('chemical', chem_id)
            if data:
                for ref in data.get('data', {}).get('crossReferences', []):
                    if ref.get('resource') == 'PubChem Compound':
                        search_term = ref.get('resourceId')
                        break
                if not search_term: # Fallback to name
                    search_term = data.get('data', {}).get('name')
                    search_type = 'name'
        else: search_term, search_type = chem_id, 'auto'

        if not search_term:
            self.stats['smiles_failures'] += 1; return None

        try:
            if search_type in ['cid', 'auto']:
                r = self.session.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{search_term}/property/CanonicalSMILES/TXT")
                if r.status_code == 200: smiles = r.text.strip()
            if not smiles and search_type in ['name', 'auto']:
                r = self.session.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{search_term}/property/CanonicalSMILES/TXT")
                if r.status_code == 200: smiles = r.text.strip()
            time.sleep(0.2)
        except Exception as e: self.last_error = f"PubChem query failed: {e}"

        if smiles and Chem.MolFromSmiles(smiles):
            with open(cache_path, "w") as f: f.write(smiles)
            return smiles

        self.stats['smiles_failures'] += 1; return None

    def fetch_sequence(self, gene_id: str) -> Optional[str]:
        gene_id = str(gene_id).strip()
        cache_path = self.seq_cache / f"{gene_id}.txt"
        if cache_path.exists():
            self.stats['seq_cache_hits'] += 1
            with open(cache_path, "r") as f: return f.read().strip()

        self.stats['seq_api_calls'] += 1
        seq, search_term, search_type = "", None, 'accession'

        if gene_id.startswith('PA'):
            data = self._get_pharmgkb_data('gene', gene_id)
            if data:
                for ref in data.get('data', {}).get('crossReferences', []):
                    if ref.get('resource') == 'UniProtKB':
                        search_term = ref.get('resourceId')
                        break
                if not search_term: # Fallback to symbol
                    search_term = data.get('data', {}).get('symbol')
                    search_type = 'gene'
        else: search_term, search_type = gene_id, 'auto'

        if not search_term:
            self.stats['seq_failures'] += 1; return None

        try:
            if search_type in ['accession', 'auto']:
                r = self.session.get(f"https://www.uniprot.org/uniprot/{search_term}.fasta")
                if r.status_code == 200: seq = "".join(r.text.split('\n')[1:])
            if not seq and search_type in ['gene', 'auto']:
                r = self.session.get(f"https://rest.uniprot.org/uniprotkb/search?query=gene:{search_term}&format=fasta&size=1")
                if r.status_code == 200 and '>' in r.text: seq = "".join(r.text.split('\n')[1:])
            time.sleep(0.2)
        except Exception as e: self.last_error = f"UniProt query failed: {e}"

        if seq and len(seq) > 20:
            with open(cache_path, "w") as f: f.write(seq)
            return seq

        self.stats['seq_failures'] += 1; return None
