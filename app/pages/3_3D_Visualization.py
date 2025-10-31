"""
3D Visualization Page - Interactive molecular structure viewer
"""

import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import CDSS modules
try:
    from utils.exceptions import CDSSException
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure all dependencies are installed.")
    st.stop()

st.set_page_config(
    page_title="3D Molecular Visualization",
    page_icon="🧬",
    layout="wide"
)

def check_imports():
    """Check if visualization libraries are available."""
    try:
        import py3Dmol
        from utils.stpy3mol_local import showmol
        return py3Dmol, showmol
    except ImportError as e:
        st.error(
            f"Visualization libraries missing: {e}\n\n"
            "Please install py3Dmol: `pip install py3Dmol`"
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
    Render RDKit molecule in 3D using py3Dmol within Streamlit.
    
    Args:
        mol: RDKit molecule object
        label: Label for the molecule
        style: Rendering style (stick, line, sphere, cartoon, ribbon)
        bg_color: Background color
        width: Viewer width
        height: Viewer height
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
            st.error("Failed to generate PDB block. Molecule may lack 3D coordinates.")
            return
    except Exception as e:
        st.error(f"Failed to convert molecule to PDB format: {e}")
        return
    
    # Set style dictionary for py3Dmol
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
    
    # Show in Streamlit
    st.write(f"**3D Structure: {label}** (style: `{style}`)")
    showmol(view, height=height, width=width)

# Main UI
st.title("🧬 3D Molecular Visualization")
st.markdown("""
Visualize drug and protein structures in interactive 3D.
Enter SMILES strings or fetch molecules from the database.
""")

# Check for core processor
if "core_processor" not in st.session_state:
    st.warning(
        "⚠️ Core processor not initialized. Some features may be limited.\n\n"
        "For full functionality, initialize the system on the main page."
    )

# Tabs for different input methods
tab1, tab2 = st.tabs(["🔬 From SMILES", "🔍 From Database"])

with tab1:
    st.subheader("Visualize Molecule from SMILES")
    
    # SMILES input
    smiles_input = st.text_input(
        "Enter SMILES String",
        placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O (Aspirin)",
        help="Enter a valid SMILES string for the molecule you want to visualize"
    )
    
    # Visualization options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        style = st.selectbox(
            "Rendering Style",
            ["stick", "sphere", "line", "cartoon", "ribbon"],
            index=0
        )
    
    with col2:
        bg_color = st.selectbox(
            "Background Color",
            ["white", "black", "gray"],
            index=0
        )
    
    with col3:
        optimize = st.checkbox(
            "Optimize 3D Structure",
            value=True,
            help="Generate optimized 3D conformer (recommended)"
        )
    
    if st.button("🚀 Visualize Molecule", type="primary"):
        if not smiles_input:
            st.warning("Please enter a SMILES string")
        else:
            with st.spinner("Generating 3D structure..."):
                try:
                    # Parse SMILES
                    mol = Chem.MolFromSmiles(smiles_input)
                    if mol is None:
                        st.error(f"Invalid SMILES string: {smiles_input}")
                        st.stop()
                    
                    # Add hydrogens
                    mol = Chem.AddHs(mol)
                    
                    # Generate 3D conformer
                    if optimize:
                        params = AllChem.ETKDGv3()
                        params.randomSeed = 42
                        
                        result = AllChem.EmbedMolecule(mol, params)
                        if result != 0:
                            st.error("Failed to generate 3D conformer. Try a simpler molecule.")
                            st.stop()
                        
                        # Optimize with force field
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                    
                    # Display molecular properties
                    st.markdown("### 📊 Molecular Properties")
                    prop_col1, prop_col2, prop_col3 = st.columns(3)
                    
                    with prop_col1:
                        st.metric("Number of Atoms", mol.GetNumAtoms())
                    
                    with prop_col2:
                        st.metric("Number of Bonds", mol.GetNumBonds())
                    
                    with prop_col3:
                        mw = Chem.Descriptors.MolWt(mol)
                        st.metric("Molecular Weight", f"{mw:.2f}")
                    
                    # Render 3D visualization
                    st.markdown("### 🧬 3D Structure")
                    render_3d_mol(
                        mol,
                        label=smiles_input[:30] + "..." if len(smiles_input) > 30 else smiles_input,
                        style=style,
                        bg_color=bg_color,
                        width=800,
                        height=500
                    )
                    
                    # Display SMILES info
                    with st.expander("📝 Molecular Details"):
                        st.markdown(f"**Input SMILES:** `{smiles_input}`")
                        canonical = Chem.MolToSmiles(mol)
                        st.markdown(f"**Canonical SMILES:** `{canonical}`")
                        
                        # Additional properties
                        st.markdown("**Descriptors:**")
                        st.write(f"- LogP: {Chem.Descriptors.MolLogP(mol):.2f}")
                        st.write(f"- H-Bond Donors: {Chem.Descriptors.NumHDonors(mol)}")
                        st.write(f"- H-Bond Acceptors: {Chem.Descriptors.NumHAcceptors(mol)}")
                        st.write(f"- Rotatable Bonds: {Chem.Descriptors.NumRotatableBonds(mol)}")
                
                except Exception as e:
                    st.error(f"Visualization failed: {e}")
                    st.exception(e)

with tab2:
    st.subheader("Visualize Molecule from Database")
    
    if "core_processor" not in st.session_state:
        st.info(
            "Database lookup requires system initialization.\n\n"
            "Please initialize the core processor on the main page."
        )
    else:
        # Drug/Gene selection
        col1, col2 = st.columns(2)
        
        with col1:
            drug_name = st.text_input(
                "Drug Name or ID",
                placeholder="e.g., Aspirin, CID 2244",
                help="Enter drug name or PubChem CID"
            )
        
        with col2:
            gene_name = st.text_input(
                "Gene Name (Optional)",
                placeholder="e.g., CYP2D6",
                help="Enter gene symbol for protein visualization"
            )
        
        # Visualization options
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            style_db = st.selectbox(
                "Rendering Style",
                ["stick", "sphere", "line"],
                index=0,
                key="style_db"
            )
        
        with viz_col2:
            bg_color_db = st.selectbox(
                "Background Color",
                ["white", "black", "gray"],
                index=0,
                key="bg_db"
            )
        
        if st.button("🔍 Fetch and Visualize", type="primary"):
            if not drug_name:
                st.warning("Please enter a drug name or ID")
            else:
                with st.spinner(f"Fetching molecular data for {drug_name}..."):
                    try:
                        # Fetch SMILES from database
                        core_processor = st.session_state.core_processor
                        smiles = core_processor.get_smiles_sync(drug_name)
                        
                        if not smiles:
                            st.error(f"Could not find SMILES for: {drug_name}")
                            st.stop()
                        
                        st.success(f"✅ Found SMILES: `{smiles[:50]}...`")
                        
                        # Generate 3D structure
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            st.error("Invalid SMILES returned from database")
                            st.stop()
                        
                        mol = Chem.AddHs(mol)
                        
                        # Embed and optimize
                        params = AllChem.ETKDGv3()
                        params.randomSeed = 42
                        result = AllChem.EmbedMolecule(mol, params)
                        
                        if result == 0:
                            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                            
                            # Render visualization
                            st.markdown("### 🧬 3D Structure")
                            render_3d_mol(
                                mol,
                                label=drug_name,
                                style=style_db,
                                bg_color=bg_color_db,
                                width=800,
                                height=500
                            )
                        else:
                            st.error("Failed to generate 3D conformer")
                    
                    except Exception as e:
                        st.error(f"Failed to fetch and visualize: {e}")
                        st.exception(e)

# Help section
st.markdown("---")
with st.expander("ℹ️ How to Use This Tool"):
    st.markdown("""
    **Method 1: From SMILES**
    1. Enter a valid SMILES string
    2. Choose visualization style and background
    3. Click "Visualize Molecule"
    4. Interact with the 3D viewer (rotate, zoom)
    
    **Method 2: From Database**
    1. Initialize the system on the main page
    2. Enter a drug name or PubChem CID
    3. Click "Fetch and Visualize"
    4. The system will automatically fetch and display the structure
    
    **Supported Styles:**
    - **Stick**: Traditional ball-and-stick model (recommended)
    - **Sphere**: Space-filling model (van der Waals radii)
    - **Line**: Simple wireframe model
    - **Cartoon**: Protein secondary structure (for proteins)
    - **Ribbon**: Protein backbone trace (for proteins)
    
    **Tips:**
    - Use mouse to rotate the structure
    - Scroll to zoom in/out
    - Click atoms to see labels
    """)

st.caption("🧬 Powered by RDKit and py3Dmol")
