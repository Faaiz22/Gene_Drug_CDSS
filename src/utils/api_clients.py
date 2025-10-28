# src/utils/api_clients.py
"""
DataEnricher: asynchronous clients to fetch chemical SMILES and protein sequences,
with local file-based caching, rate-limiting and robust error handling.

This is a cleaned, re-structured and debugged replacement for the previously
fragmented implementation. Public methods fetch_smiles and fetch_sequence return
a string on success and raise DataFetchException with a formatted user message
on failure.
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


class DataEnricher:
    """
    Fetches protein sequences and chemical SMILES from external APIs asynchronously.
    Caches results locally for efficiency.
    Usage:
        async with DataEnricher(config) as enricher:
            smiles = await enricher.fetch_smiles("1234")
            seq = await enricher.fetch_sequence("P12345")
    """

    def __init__(self, config: Dict):
        self.config = config

        cache_dir = Path(self.config.get("paths", {}).get("cache_dir", "/tmp/drug_gene_cache"))
        self.smiles_cache = cache_dir / "smiles"
        self.seq_cache = cache_dir / "sequences"
        os.makedirs(self.smiles_cache, exist_ok=True)
        os.makedirs(self.seq_cache, exist_ok=True)

        headers = {"User-Agent": "DrugGeneCDSS/1.0"}
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        self.pharmgkb_api = "https://api.pharmgkb.org/v1/data"
        self.stats = {
            "smiles_cache_hits": 0,
            "smiles_api_calls": 0,
            "smiles_failures": 0,
            "seq_cache_hits": 0,
            "seq_api_calls": 0,
            "seq_failures": 0,
        }
        self.last_error = ""
        self._request_lock = asyncio.Lock()
        self._last_request_time = 0.0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def _rate_limit(self, delay: float = 0.34):
        """Enforce rate limiting (~3 requests/sec by default)."""
        async with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < delay:
                await asyncio.sleep(delay - time_since_last)
            self._last_request_time = time.time()

    def _get_cache_key(self, identifier: str) -> str:
        """Generate deterministic cache key (hex md5)."""
        return hashlib.md5(identifier.encode("utf-8")).hexdigest()

    async def _get_pharmgkb_data(self, entity_type: str, entity_id: str) -> Optional[dict]:
        """
        Fetch data from PharmGKB API with retries and exponential backoff.
        Returns parsed JSON on success, None otherwise (last_error is set).
        """
        url = f"{self.pharmgkb_api}/{entity_type}/{entity_id}"
        for attempt in range(3):
            try:
                await self._rate_limit()
                r = await self.client.get(url)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    # rate limited, backoff and retry
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    self.last_error = f"PharmGKB API error {r.status_code} for {entity_type} {entity_id}"
                    break
            except httpx.TimeoutException:
                self.last_error = f"Timeout querying PharmGKB for {entity_id}"
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
            except httpx.RequestError as e:
                self.last_error = f"PharmGKB query failed for {entity_id}: {e}"
                break
            except Exception as e:
                self.last_error = f"Unexpected error querying PharmGKB for {entity_id}: {e}"
                break
        return None

    async def fetch_smiles(self, chem_id: str) -> str:
        """
        Fetch SMILES for a chemical identifier.

        On success returns a validated SMILES string.
        On failure raises DataFetchException with a formatted user-facing message.
        """
        chem_id = str(chem_id or "").strip()
        if not chem_id:
            raise DataFetchException(
                format_user_error("smiles_not_found", drug_id=chem_id),
                "Empty or whitespace-only identifier provided",
            )

        cache_key = self._get_cache_key(chem_id)
        cache_path = self.smiles_cache / f"{cache_key}.json"

        # Check cache
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                smiles = data.get("smiles")
                if smiles and Chem.MolFromSmiles(smiles):
                    self.stats["smiles_cache_hits"] += 1
                    return smiles
            except (json.JSONDecodeError, KeyError):
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass  # ignore cache cleanup failures

        self.stats["smiles_api_calls"] += 1

        # Determine search term and type
        search_term: Optional[str] = None
        search_type = "auto"
        if chem_id.upper().startswith("PA"):
            pharm = await self._get_pharmgkb_data("chemical", chem_id)
            if pharm:
                for ref in pharm.get("data", {}).get("crossReferences", []):
                    if ref.get("resource") == "PubChem Compound":
                        search_term = ref.get("resourceId")
                        search_type = "cid"
                        break
                if not search_term:
                    search_term = pharm.get("data", {}).get("name")
                    search_type = "name"
        else:
            search_term = chem_id
            search_type = "auto"

        if not search_term:
            self.stats["smiles_failures"] += 1
            raise DataFetchException(
                format_user_error("smiles_not_found", drug_id=chem_id),
                f"Could not determine search term for {chem_id}. Last error: {self.last_error}",
            )

        pubchem_urls = []
        if search_type in ("cid", "auto"):
            pubchem_urls.append(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{search_term}/property/CanonicalSMILES/TXT"
            )
        if search_type in ("name", "auto"):
            pubchem_urls.append(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{search_term}/property/CanonicalSMILES/TXT"
            )

        smiles: Optional[str] = None
        for url in pubchem_urls:
            for attempt in range(3):
                try:
                    await self._rate_limit()
                    r = await self.client.get(url)
                    if r.status_code == 200:
                        candidate = r.text.strip()
                        if candidate and Chem.MolFromSmiles(candidate):
                            smiles = candidate
                            # cache
                            try:
                                with open(cache_path, "w", encoding="utf-8") as f:
                                    json.dump(
                                        {"smiles": smiles, "original_id": chem_id, "timestamp": time.time()},
                                        f,
                                    )
                            except Exception:
                                # cache failures shouldn't block success
                                pass
                            return smiles
                        else:
                            # invalid SMILES from PubChem (unlikely) - record error and try next
                            self.last_error = f"Invalid SMILES received from PubChem for {search_term}"
                            break
                    elif r.status_code == 404:
                        # not found, try next url
                        break
                    elif r.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        self.last_error = f"PubChem returned {r.status_code} for URL {url}"
                        break
                except httpx.TimeoutException:
                    if attempt < 2:
                        await asyncio.sleep(1)
                        continue
                    self.last_error = f"Timeout querying PubChem for {search_term}"
                except httpx.RequestError as e:
                    self.last_error = f"PubChem request failed: {e}"
                    break
                except Exception as e:
                    self.last_error = f"Unexpected error querying PubChem: {e}"
                    break

        self.stats["smiles_failures"] += 1
        raise DataFetchException(
            format_user_error("smiles_not_found", drug_id=chem_id),
            f"SMILES not found for {chem_id}. Last error: {self.last_error}",
        )

    async def fetch_sequence(self, gene_id: str) -> str:
        """
        Fetch protein sequence for a gene identifier.

        On success returns the sequence string (AA letters).
        On failure raises DataFetchException with a formatted user-facing message.
        """
        gene_id = str(gene_id or "").strip()
        if not gene_id:
            raise DataFetchException(
                format_user_error("sequence_not_found", gene_id=gene_id),
                "Empty or whitespace-only identifier provided",
            )

        cache_key = self._get_cache_key(gene_id)
        cache_path = self.seq_cache / f"{cache_key}.json"

        # Check cache
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                seq = data.get("sequence")
                if seq and len(seq) > 20:
                    self.stats["seq_cache_hits"] += 1
                    return seq
            except (json.JSONDecodeError, KeyError):
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass

        self.stats["seq_api_calls"] += 1

        # Determine search term and type
        search_term: Optional[str] = None
        search_type = "auto"
        if gene_id.upper().startswith("PA"):
            pharm = await self._get_pharmgkb_data("gene", gene_id)
            if pharm:
                for ref in pharm.get("data", {}).get("crossReferences", []):
                    if ref.get("resource") == "UniProtKB":
                        search_term = ref.get("resourceId")
                        search_type = "accession"
                        break
                if not search_term:
                    # fallback to gene symbol
                    search_term = pharm.get("data", {}).get("symbol")
                    search_type = "gene"
        else:
            search_term = gene_id
            search_type = "auto"

        if not search_term:
            self.stats["seq_failures"] += 1
            raise DataFetchException(
                format_user_error("sequence_not_found", gene_id=gene_id),
                f"Could not determine UniProt search term for {gene_id}. Last error: {self.last_error}",
            )

        uniprot_urls = []
        if search_type in ("accession", "auto"):
            uniprot_urls.append(f"https://rest.uniprot.org/uniprotkb/{search_term}.fasta")
        if search_type in ("gene", "auto"):
            # query by gene symbol, return one result (size=1)
            uniprot_urls.append(
                f"https://rest.uniprot.org/uniprotkb/search?query=gene:{search_term}&format=fasta&size=1"
            )

        seq: Optional[str] = None
        for url in uniprot_urls:
            for attempt in range(3):
                try:
                    await self._rate_limit()
                    r = await self.client.get(url)
                    if r.status_code == 200 and r.text and ">" in r.text:
                        lines = r.text.splitlines()
                        # skip header lines starting with '>'
                        seq_lines = [ln.strip() for ln in lines if not ln.startswith(">")]
                        candidate = "".join(seq_lines)
                        if candidate and len(candidate) > 20:
                            seq = candidate
                            # cache
                            try:
                                with open(cache_path, "w", encoding="utf-8") as f:
                                    json.dump(
                                        {"sequence": seq, "original_id": gene_id, "timestamp": time.time()},
                                        f,
                                    )
                            except Exception:
                                pass
                            return seq
                        else:
                            self.last_error = f"Sequence too short or invalid from UniProt for {search_term}"
                            break
                    elif r.status_code == 404:
                        break
                    elif r.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        self.last_error = f"UniProt returned {r.status_code} for URL {url}"
                        break
                except httpx.TimeoutException:
                    if attempt < 2:
                        await asyncio.sleep(1)
                        continue
                    self.last_error = f"Timeout querying UniProt for {search_term}"
                except httpx.RequestError as e:
                    self.last_error = f"UniProt request failed: {e}"
                    break
                except Exception as e:
                    self.last_error = f"Unexpected error querying UniProt: {e}"
                    break

        self.stats["seq_failures"] += 1
        raise DataFetchException(
            format_user_error("sequence_not_found", gene_id=gene_id),
            f"Sequence not found for {gene_id}. Last error: {self.last_error}",
        )
