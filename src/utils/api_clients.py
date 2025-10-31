"""
Enhanced API Clients with PubChemClient and UniProtClient.
Provides asynchronous data fetching with caching and error handling.
"""

import os
import time
import json
import asyncio
import hashlib
from pathlib import Path
from typing import Optional, Dict

import httpx
from rdkit import Chem

from .exceptions import DataFetchException, format_user_error


class PubChemClient:
    """
    Dedicated client for PubChem API interactions.
    Fetches chemical structures (SMILES) from compound names or CIDs.
    """
    
    def __init__(self, base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "DrugGeneCDSS/1.0"}
        )
        self._request_lock = asyncio.Lock()
        self._last_request_time = 0.0
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _rate_limit(self, delay: float = 0.34):
        """Enforce rate limiting (~3 requests/sec)."""
        async with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < delay:
                await asyncio.sleep(delay - time_since_last)
            self._last_request_time = time.time()
    
    async def get_smiles_from_name(self, compound_name: str) -> Optional[str]:
        """Fetch SMILES from compound name."""
        url = f"{self.base_url}/compound/name/{compound_name}/property/CanonicalSMILES/TXT"
        
        for attempt in range(3):
            try:
                await self._rate_limit()
                response = await self.client.get(url)
                
                if response.status_code == 200:
                    smiles = response.text.strip()
                    if smiles and Chem.MolFromSmiles(smiles):
                        return smiles
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                elif response.status_code == 404:
                    return None
                    
            except Exception as e:
                if attempt == 2:
                    raise DataFetchException(
                        f"Failed to fetch SMILES for {compound_name}",
                        str(e)
                    )
                await asyncio.sleep(1)
        
        return None
    
    async def get_smiles_from_cid(self, cid: str) -> Optional[str]:
        """Fetch SMILES from PubChem CID."""
        url = f"{self.base_url}/compound/cid/{cid}/property/CanonicalSMILES/TXT"
        
        for attempt in range(3):
            try:
                await self._rate_limit()
                response = await self.client.get(url)
                
                if response.status_code == 200:
                    smiles = response.text.strip()
                    if smiles and Chem.MolFromSmiles(smiles):
                        return smiles
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                elif response.status_code == 404:
                    return None
                    
            except Exception as e:
                if attempt == 2:
                    raise DataFetchException(
                        f"Failed to fetch SMILES for CID {cid}",
                        str(e)
                    )
                await asyncio.sleep(1)
        
        return None


class UniProtClient:
    """
    Dedicated client for UniProt API interactions.
    Fetches protein sequences from gene symbols or UniProt accessions.
    """
    
    def __init__(self, base_url: str = "https://rest.uniprot.org/uniprotkb"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "DrugGeneCDSS/1.0"}
        )
        self._request_lock = asyncio.Lock()
        self._last_request_time = 0.0
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _rate_limit(self, delay: float = 0.34):
        """Enforce rate limiting."""
        async with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < delay:
                await asyncio.sleep(delay - time_since_last)
            self._last_request_time = time.time()
    
    async def get_sequence_from_query(self, query: str) -> Optional[str]:
        """
        Fetch protein sequence from UniProt query.
        Query format: "gene:BRCA1 AND organism:'homo sapiens'"
        """
        url = f"{self.base_url}/search"
        params = {
            "query": query,
            "format": "fasta",
            "size": 1
        }
        
        for attempt in range(3):
            try:
                await self._rate_limit()
                response = await self.client.get(url, params=params)
                
                if response.status_code == 200 and ">" in response.text:
                    lines = response.text.splitlines()
                    seq_lines = [ln.strip() for ln in lines if not ln.startswith(">")]
                    sequence = "".join(seq_lines)
                    
                    if sequence and len(sequence) > 20:
                        return sequence
                        
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                elif response.status_code == 404:
                    return None
                    
            except Exception as e:
                if attempt == 2:
                    raise DataFetchException(
                        f"Failed to fetch sequence for query: {query}",
                        str(e)
                    )
                await asyncio.sleep(1)
        
        return None
    
    async def get_sequence_from_accession(self, accession: str) -> Optional[str]:
        """Fetch protein sequence from UniProt accession."""
        url = f"{self.base_url}/{accession}.fasta"
        
        for attempt in range(3):
            try:
                await self._rate_limit()
                response = await self.client.get(url)
                
                if response.status_code == 200 and ">" in response.text:
                    lines = response.text.splitlines()
                    seq_lines = [ln.strip() for ln in lines if not ln.startswith(">")]
                    sequence = "".join(seq_lines)
                    
                    if sequence and len(sequence) > 20:
                        return sequence
                        
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                elif response.status_code == 404:
                    return None
                    
            except Exception as e:
                if attempt == 2:
                    raise DataFetchException(
                        f"Failed to fetch sequence for accession {accession}",
                        str(e)
                    )
                await asyncio.sleep(1)
        
        return None


class DataEnricher:
    """
    High-level enricher that combines PubChem and UniProt clients.
    Provides unified interface with caching for the CDSS system.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        cache_dir = Path(config.get("paths", {}).get("cache_dir", "/tmp/drug_gene_cache"))
        self.smiles_cache = cache_dir / "smiles"
        self.seq_cache = cache_dir / "sequences"
        os.makedirs(self.smiles_cache, exist_ok=True)
        os.makedirs(self.seq_cache, exist_ok=True)
        
        # Initialize clients
        pubchem_url = config.get("api", {}).get("pubchem_base_url", 
                                                  "https://pubchem.ncbi.nlm.nih.gov/rest/pug")
        uniprot_url = config.get("api", {}).get("uniprot_base_url",
                                                  "https://rest.uniprot.org/uniprotkb")
        
        self.pubchem = PubChemClient(pubchem_url)
        self.uniprot = UniProtClient(uniprot_url)
        
        self.stats = {
            "smiles_cache_hits": 0,
            "smiles_api_calls": 0,
            "smiles_failures": 0,
            "seq_cache_hits": 0,
            "seq_api_calls": 0,
            "seq_failures": 0,
        }
        self.last_error = ""
    
    async def __aenter__(self):
        await self.pubchem.__aenter__()
        await self.uniprot.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.pubchem.__aexit__(exc_type, exc_val, exc_tb)
        await self.uniprot.__aexit__(exc_type, exc_val, exc_tb)
    
    def _get_cache_key(self, identifier: str) -> str:
        """Generate deterministic cache key."""
        return hashlib.md5(identifier.encode("utf-8")).hexdigest()
    
    async def fetch_smiles(self, chem_id: str) -> str:
        """
        Fetch SMILES for a chemical identifier with caching.
        Raises DataFetchException on failure.
        """
        chem_id = str(chem_id or "").strip()
        if not chem_id:
            raise DataFetchException(
                format_user_error("smiles_not_found", drug_id=chem_id),
                "Empty identifier provided"
            )
        
        # Check cache
        cache_key = self._get_cache_key(chem_id)
        cache_path = self.smiles_cache / f"{cache_key}.json"
        
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                smiles = data.get("smiles")
                if smiles and Chem.MolFromSmiles(smiles):
                    self.stats["smiles_cache_hits"] += 1
                    return smiles
            except (json.JSONDecodeError, KeyError):
                cache_path.unlink(missing_ok=True)
        
        self.stats["smiles_api_calls"] += 1
        
        # Try fetching from PubChem
        smiles = None
        
        # Try as CID first if numeric
        if chem_id.isdigit():
            try:
                smiles = await self.pubchem.get_smiles_from_cid(chem_id)
            except Exception as e:
                self.last_error = str(e)
        
        # Try as compound name
        if not smiles:
            try:
                smiles = await self.pubchem.get_smiles_from_name(chem_id)
            except Exception as e:
                self.last_error = str(e)
        
        if not smiles:
            self.stats["smiles_failures"] += 1
            raise DataFetchException(
                format_user_error("smiles_not_found", drug_id=chem_id),
                f"SMILES not found for {chem_id}. Last error: {self.last_error}"
            )
        
        # Cache result
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "smiles": smiles,
                    "original_id": chem_id,
                    "timestamp": time.time()
                }, f)
        except Exception:
            pass  # Cache failures shouldn't block success
        
        return smiles
    
    async def fetch_sequence(self, gene_id: str) -> str:
        """
        Fetch protein sequence for a gene identifier with caching.
        Raises DataFetchException on failure.
        """
        gene_id = str(gene_id or "").strip()
        if not gene_id:
            raise DataFetchException(
                format_user_error("sequence_not_found", gene_id=gene_id),
                "Empty identifier provided"
            )
        
        # Check cache
        cache_key = self._get_cache_key(gene_id)
        cache_path = self.seq_cache / f"{cache_key}.json"
        
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                seq = data.get("sequence")
                if seq and len(seq) > 20:
                    self.stats["seq_cache_hits"] += 1
                    return seq
            except (json.JSONDecodeError, KeyError):
                cache_path.unlink(missing_ok=True)
        
        self.stats["seq_api_calls"] += 1
        
        # Try fetching from UniProt
        sequence = None
        
        # Try as accession first if it looks like one
        if gene_id[0].isalpha() and len(gene_id) >= 6:
            try:
                sequence = await self.uniprot.get_sequence_from_accession(gene_id)
            except Exception as e:
                self.last_error = str(e)
        
        # Try as gene symbol
        if not sequence:
            try:
                organism = self.config.get("api", {}).get("uniprot_organism", "homo sapiens")
                query = f"gene:{gene_id} AND organism:\"{organism}\""
                sequence = await self.uniprot.get_sequence_from_query(query)
            except Exception as e:
                self.last_error = str(e)
        
        if not sequence:
            self.stats["seq_failures"] += 1
            raise DataFetchException(
                format_user_error("sequence_not_found", gene_id=gene_id),
                f"Sequence not found for {gene_id}. Last error: {self.last_error}"
            )
        
        # Cache result
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "sequence": sequence,
                    "original_id": gene_id,
                    "timestamp": time.time()
                }, f)
        except Exception:
            pass
        
        return sequence
