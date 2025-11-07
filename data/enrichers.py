import hashlib
import json
from pathlib import Path
from typing import Optional
import requests
from Bio import Entrez

class NCBISequenceFetcher:
    """
    Fetches protein sequences from NCBI with disk caching.
    """
    
    def __init__(self, cache_dir: Path, email: str = "your@email.com"):
        self.cache_dir = cache_dir / "sequences"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        Entrez.email = email
    
    def get_sequence(self, gene_id: str) -> Optional[str]:
        """
        Fetch sequence with caching.
        """
        cache_key = hashlib.md5(str(gene_id).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.txt"
        
        # Check cache
        if cache_path.exists():
            return cache_path.read_text()
        
        # Fetch from NCBI
        try:
            # Use NCBI Gene -> Protein link
            handle = Entrez.efetch(
                db="gene",
                id=gene_id,
                rettype="fasta_cds_na",
                retmode="text"
            )
            sequence = self._parse_fasta(handle.read())
            
            if sequence:
                cache_path.write_text(sequence)
                return sequence
        except Exception as e:
            logging.error(f"Failed to fetch sequence for gene {gene_id}: {e}")
        
        return None
    
    def _parse_fasta(self, fasta_text: str) -> Optional[str]:
        """Extract sequence from FASTA format."""
        lines = fasta_text.strip().split('\n')
        if not lines or not lines[0].startswith('>'):
            return None
        return ''.join(lines[1:])
