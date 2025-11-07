import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional
from rdkit import Chem
from .enrichers import NCBISequenceFetcher, ChemicalValidator

class UnifiedDatasetLoader:
    """
    Single entry point for loading and validating the unified dataset.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_path = Path(config['paths']['unified_dataset'])
        self.cache_dir = Path(config['paths']['cache_dir'])
        self.logger = logging.getLogger(__name__)
        
        # Initialize enrichers
        self.ncbi_fetcher = NCBISequenceFetcher(cache_dir=self.cache_dir)
        self.chem_validator = ChemicalValidator()
    
    def load_and_validate(self) -> pd.DataFrame:
        """
        Load dataset with validation and automatic enrichment.
        """
        self.logger.info(f"Loading dataset from {self.data_path}")
        
        # 1. Load CSV
        df = pd.read_csv(self.data_path)
        initial_count = len(df)
        
        # 2. Validate required columns
        required_cols = ['pharmgkb_gene_id', 'ncbi_gene_id', 'pharmgkb_drug_id', 
                        'smiles', 'label']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # 3. Validate SMILES
        df['smiles_valid'] = df['smiles'].apply(self._validate_smiles)
        invalid_smiles = (~df['smiles_valid']).sum()
        if invalid_smiles > 0:
            self.logger.warning(f"Found {invalid_smiles} invalid SMILES strings")
            df = df[df['smiles_valid']].copy()
        
        # 4. Enrich with sequences (cached)
        df['protein_sequence'] = df['ncbi_gene_id'].apply(
            self._fetch_sequence
        )
        
        # 5. Remove entries without sequences
        df = df[df['protein_sequence'].notna()].copy()
        
        final_count = len(df)
        self.logger.info(
            f"Dataset loaded: {final_count}/{initial_count} valid pairs "
            f"({(final_count/initial_count)*100:.1f}%)"
        )
        
        return df
    
    def _validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _fetch_sequence(self, ncbi_gene_id: str) -> Optional[str]:
        """Fetch protein sequence from NCBI (cached)."""
        return self.ncbi_fetcher.get_sequence(ncbi_gene_id)
