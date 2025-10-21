import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from rdkit import Chem


def process_raw_data(config: dict, enricher) -> pd.DataFrame:
    """Loads, filters, enriches, and cleans the raw relationship data."""
    input_path = Path(config['paths']['data_dir']) / config['paths']['relationships_file']
    print(f"\n📁 Reading file: {input_path}")
    df_raw = pd.read_csv(input_path, sep="\t")
    print(f"✓ Loaded {len(df_raw)} total rows")

    # --- Filter for Gene-Chemical pairs ---
    print("\nSTEP 1: Filtering for Gene-Chemical pairs...")
    possible_cols = {
        'e1_type': ['Entity1_type', 'entity1_type', 'source_type'],
        'e2_type': ['Entity2_type', 'entity2_type', 'target_type'],
        'e1_id': ['Entity1_id', 'entity1_id', 'source_id'],
        'e2_id': ['Entity2_id', 'entity2_id', 'target_id'],
        'e1_name': ['Entity1_name', 'entity1_name', 'source_name'],
        'e2_name': ['Entity2_name', 'entity2_name', 'target_name']
    }
    actual_cols = {key: next((c for c in pots if c in df_raw.columns), None) for key, pots in possible_cols.items()}

    if not all(actual_cols[k] for k in ['e1_type', 'e2_type', 'e1_id', 'e2_id']):
        raise ValueError("Essential columns (ID and type for both entities) not found in relationships.tsv")

    df_gc = df_raw[(df_raw[actual_cols['e1_type']].str.lower() == 'gene') & (df_raw[actual_cols['e2_type']].str.lower() == 'chemical')].copy()
    df_cg = df_raw[(df_raw[actual_cols['e1_type']].str.lower() == 'chemical') & (df_raw[actual_cols['e2_type']].str.lower() == 'gene')].copy()

    gc_rename = {v: k.replace('e1', 'gene').replace('e2', 'chem') for k, v in actual_cols.items() if v}
    cg_rename = {v: k.replace('e1', 'chem').replace('e2', 'gene') for k, v in actual_cols.items() if v}

    df_gc_std = df_gc.rename(columns=gc_rename)
    df_cg_std = df_cg.rename(columns=cg_rename)

    final_cols = ['gene_id', 'chem_id', 'gene_name', 'chem_name']
    final_cols = [c for c in final_cols if c in df_gc_std.columns or c in df_cg_std.columns]
    if 'label' in df_raw.columns: final_cols.append('label')

    df_filtered = pd.concat([df_gc_std[final_cols], df_cg_std[final_cols]], ignore_index=True)
    df_filtered = df_filtered.dropna(subset=['gene_id', 'chem_id'])
    print(f"✓ Found {len(df_filtered)} standardized Gene-Chemical pairs.")

    # --- Add Labels if Missing ---
    if 'label' not in df_filtered.columns:
        df_filtered['label'] = 1
    df_filtered = df_filtered.dropna(subset=['label'])

    # --- Enrich with Sequences and SMILES ---
    print("\nSTEP 2: Enriching data with sequences and SMILES...")
    unique_genes = df_filtered['gene_id'].drop_duplicates()
    seq_map = {gene_id: enricher.fetch_sequence(str(gene_id)) for gene_id in tqdm(unique_genes, desc="Sequences")}
    df_filtered["sequence"] = df_filtered['gene_id'].map(seq_map)

    unique_chems = df_filtered['chem_id'].drop_duplicates()
    smiles_map = {chem_id: enricher.fetch_smiles(str(chem_id)) for chem_id in tqdm(unique_chems, desc="SMILES")}
    df_filtered["smiles"] = df_filtered['chem_id'].map(smiles_map)

    # --- Clean and Validate Data ---
    print("\nSTEP 3: Cleaning and validating enriched data...")
    initial_rows = len(df_filtered)
    df_clean = df_filtered.dropna(subset=["smiles", "sequence"]).reset_index(drop=True)
    print(f"  • Dropped {initial_rows - len(df_clean)} rows with missing data.")

    valid_indices = [i for i, smi in enumerate(df_clean['smiles']) if Chem.MolFromSmiles(smi) is not None]
    if len(valid_indices) < len(df_clean):
        print(f"  • Dropped {len(df_clean) - len(valid_indices)} rows with invalid SMILES.")
        df_clean = df_clean.iloc[valid_indices].reset_index(drop=True)

    print(f"✓ Data processing complete. Final valid pairs: {len(df_clean)}")
    return df_clean
