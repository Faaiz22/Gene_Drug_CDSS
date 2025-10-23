# This file serves as a clean interface between the Streamlit app
# and the core visualization logic in the `src` directory.

# Import the core rendering function from the `src` library
from src.molecular_3d.visualization import render_3d_mol

# __all__ defines the public API of this module
__all__ = ['render_3d_mol']
