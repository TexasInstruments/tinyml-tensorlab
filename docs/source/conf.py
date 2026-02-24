# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add path to the project for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'TI Tiny ML Tensorlab'
copyright = '2026, Texas Instruments Incorporated'
author = 'Texas Instruments'
release = '1.3.0'
version = 'v1.3.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'myst_parser',                # For Markdown support
    'sphinx_copybutton',          # Copy button for code blocks
    'sphinx_tabs.tabs',           # Tabbed content (Linux/Windows)
    'sphinx_design',              # Cards, grids, dropdowns
]

# MyST parser configuration (for Markdown files)
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]
myst_heading_anchors = 3

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# Templates and static files
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_static_path = ['_static']
html_css_files = ['css/custom.css']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Logo configuration - banner at top linking to TI website
html_logo = '_static/img/ti_logo.png'
# html_favicon = '_static/img/favicon.ico'
html_show_sourcelink = False
html_show_sphinx = False

# Custom footer with copyright
html_last_updated_fmt = '%b %d, %Y'

# -- Options for LaTeX/PDF output --------------------------------------------
latex_engine = 'pdflatex'
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{charter}
\usepackage{inconsolata}
''',
}

latex_documents = [
    ('index', 'TinyMLTensorlab.tex', 'Tiny ML Tensorlab User Guide',
     'Texas Instruments', 'manual'),
]

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pytorch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Todo extension configuration --------------------------------------------
todo_include_todos = True

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autosummary_generate = True

# -- Copy button configuration -----------------------------------------------
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True

# -- Tabs configuration ------------------------------------------------------
sphinx_tabs_valid_builders = ['html']
sphinx_tabs_disable_tab_closing = True
