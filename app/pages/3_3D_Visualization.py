import streamlit as st
from rdkit import Chem
from typing import Optional
import sys
from pathlib import Path

# Add project root to path (same pattern as main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Absolute imports 
from utils.module_name import ClassName
from core_processing import CoreProcessor

# NOW import CDSS modules (using absolute imports)
try:
    from agent.agent_orchestrator import DTIAgentOrchestrator
    from utils.exceptions import CDSSException
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure all dependencies are installed and the system is properly initialized.")
    st.stop()

st.set_page_config(
    page_title="3D Visualization - Agentic CDSS",
    page_icon="🎯",
    layout="wide"
)

def check_imports():
    try:
        import py3Dmol
        from src.utils.stpy3mol_local import showmol
        return py3Dmol, showmol
    except ImportError as e:
        st.error(
            f"""Visualization libraries missing or cannot be loaded: {e}
            \nPlease ensure py3Dmol is installed (`pip install py3Dmol`) and that `src/utils/stpy3mol_local.py` is available."""
        )
        return None, None

py3Dmol, showmol = check_imports()

def render_3d_mol(
        mol: Chem.Mol, 
        label: str = "molecule", 
        style: str = "stick",
        bg_color: str = "white",
        width: int = 600, 
        height: int = 400
    ) -> None:
    """
    Renders an RDKit molecule object in 3D using py3Dmol within Streamlit.

    Args:
        mol (Chem.Mol): The RDKit molecule object.
        label (str): Label for conformer.
        style (str): Rendering style: stick, line, spheres, cartoon, ribbon.
        bg_color (str): Background color for the viewer.
        width (int): Viewer width.
        height (int): Viewer height.
    """
    if py3Dmol is None or showmol is None:
        st.warning("Unable to visualize: py3Dmol or showmol missing.")
        return

    if mol is None or mol.GetNumAtoms() == 0:
        st.error("Invalid or empty molecule provided.")
        return

    # Convert RDKit mol to PDB block
    try:
        pdb_block = Chem.MolToPDBBlock(mol)
        if not pdb_block.strip():
            st.error("Failed to generate PDB block. Molecule may be invalid or lack 3D coordinates.")
            return
    except Exception as e:
        st.error(f"Failed to convert molecule to PDB format: {e}")
        return
    
    # Set style dict for py3Dmol
    styles_dict = {
        "stick": {'stick': {}},
        "line": {'line': {}},
        "sphere": {'sphere': {}},
        "cartoon": {'cartoon': {}},
        "ribbon": {'ribbon': {}}
    }
    style_chosen = styles_dict.get(style, {'stick': {}})
    
    # Create py3Dmol view
    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor(bg_color)
    view.addModel(pdb_block, 'pdb')
    view.setStyle(style_chosen)
    view.zoomTo()
    # Interactive atom label
    view.setClickable({'atom': True, 'bond': False}, True, "javascript:alert('Atom: ' + atom.atom + ' Residue: ' + atom.resi + ' Name: ' + atom.resn);")
    
    # Show in Streamlit
    st.write(f"**3D Conformer: {label}** (style: `{style}`)")
    showmol(view, height=height, width=width)

# Example usage:
# render_3d_mol(mol, label="Caffeine", style="stick", bg_color="white")
