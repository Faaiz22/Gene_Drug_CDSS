import streamlit.components.v1 as components
import py3Dmol
import ipywidgets # This is a dependency, make sure it's in requirements.txt

def showmol(mdl, height=500, width=500):
  """
  A simple wrapper to display a py3Dmol object in Streamlit.
  """
  mdl.this.params["height"] = height
  mdl.this.params["width"] = width
  components.html(mdl.this.html(), height=height, width=width)
