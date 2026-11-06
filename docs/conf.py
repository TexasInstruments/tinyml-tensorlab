# Configuration file for the Sphinx documentation builder.
#
# The source root is this docs/ folder itself, so the Markdown guides that live
# here (CONFIG_YAML_GUIDE.md, BRING_YOUR_OWN_MODEL.md, DEPLOYING_TO_SDK_EXAMPLE.md,
# NPU_CONFIGURATION_GUIDELINES.md, ondevice_training/, integer_ondevice_training/,
# deploying_*/) are pages of the user guide and keep working on GitHub as-is.
#
# Build:  pip install -r requirements.txt && make html   (output in _build/html)

import os
import shutil

# -- Project information -----------------------------------------------------
project = 'TI Tiny ML ModelZoo'
copyright = '2026, Texas Instruments Incorporated'
author = 'Texas Instruments'
release = '1.6.0'
version = 'v1.6.0'

# -- General configuration ---------------------------------------------------
extensions = [
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

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    'README_old.md',              # pre-1.5.0 model tables, kept for reference only
    'slides/**',                  # HTML decks, copied verbatim (see setup() below)
    'graphs/**',
    'requirements.txt',
]
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

html_logo = '_static/img/ti_logo.png'
html_show_sourcelink = False
html_show_sphinx = False
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
    ('index', 'TinyMLModelZoo.tex', 'Tiny ML ModelZoo User Guide',
     'Texas Instruments', 'manual'),
]

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pytorch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

todo_include_todos = True

# The existing Markdown guides start at H1 and jump to H3; harmless in the rendered site.
suppress_warnings = ['myst.header']


# -- Copy the HTML slide decks next to the built pages -----------------------
# The Markdown guides link to slides/<deck>.html; keep that path valid in the
# built site too.
def _copy_slides(app, exception):
    if exception is not None or app.builder.format != 'html':
        return
    src = os.path.join(app.srcdir, 'slides')
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(app.outdir, 'slides'), dirs_exist_ok=True)


def setup(app):
    app.connect('build-finished', _copy_slides)
