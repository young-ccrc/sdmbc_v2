import os
import sys
from unittest.mock import MagicMock

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath(".."))

# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SDMBC_v2"
copyright = "2024, Youngil Kim"
author = "Youngil Kim"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # For automatically generating documentation from docstrings
    "sphinx.ext.napoleon",  # For supporting Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Adds links to the source code
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock libraries that are difficult to install or unnecessary for documentation
MOCK_MODULES = ["xarray", "dask", "yaml", "cartopy", "yaml", "cartopy"]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the theme to make it look professional, similar to Read the Docs
html_theme = "sphinx_rtd_theme"
html_static_path = [sphinx_rtd_theme.get_html_theme_path()]
