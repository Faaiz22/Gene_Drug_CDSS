import streamlit as st
from rdkit import Chem
from typing import Optional

try:
    import py3Dmol
    from src.utils.stpy3mol_local import showmol 
except ImportError:
    py3Dmol = None
    showmol = None

def render_3d_mol(mol: Chem.Mol, label: str = "molecule") -> None:
    """
    Renders an RDKit molecule object in 3D using py3Dmol within Streamlit.
    """
    if py3Dmol is None or showmol is None:
        st.error("Visualization libraries not found. Please run: pip install py3Dmol stpy3mol")
        return

    # Convert RDKit mol to PDB block
    try:
        pdb_block = Chem.MolToPDBBlock(mol)
    except Exception as e:
        st.error(f"Failed to convert molecule to PDB format: {e}")
        return

    # Create py3Dmol view
    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'stick': {}})
    view.zoomTo()
    view.setClickable({'atom': True, 'bond': False}, True, f"javascript:alert('Atom: ' + atom.atom + ' ' + atom.resi + ' ' + atom.resn)")

    # Show in Streamlit
    st.write(f"**3D Conformer: {label}**")
    showmol(view, height=400, width=600)

# You can add more complex visualization functions here later,
# for example, to show the protein-ligand complex.
