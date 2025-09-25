# Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# -- Path setup --------------------------------------------------------------

import datetime
import os
import json
import sys
from pathlib import Path
from sphinx import __version__ as sphinx_version

# -- Project information -----------------------------------------------------

project = 'Celeritas'
all_authors = [
 'Seth R Johnson, *editor*',
 'The Celeritas Team',
]
author = " and ".join(all_authors)
copyright = '{:%Y}, UT–Battelle/ORNL and Celeritas team'.format(
    datetime.datetime.today()
)

try:
    build_dir = Path(os.environ['CMAKE_CURRENT_BINARY_DIR'])
    with open(build_dir / 'config.json', 'r') as f:
        celer_config = json.load(f)
except (KeyError, IOError) as e:
    print("Failed to load config data:", e)
    build_dir = '.'
    furo = True
    try:
        import furo
    except ImportError:
        furo = False

    celer_config = {
        "version": "*unknown version*",
        "release": "*unknown release*",
        "options": {
            "breathe": False,
            "furo": furo,
            "sphinxbib": False,
        },
        "executables": {},
    }
    # Note: 'tags' is available in `locals()` thanks to Sphinx
    tags.add('noconfig')

version = celer_config['version']
release = celer_config['release']
html_title = f"Celeritas {version} documentation"

# Set nobreathe, sphinxbib, etc for use in 'only' directives.
for (opt, val) in celer_config['options'].items():
    prefix = '' if val else 'no'
    tags.add(prefix + opt)


def setup(app):
    """Register custom directives when Sphinx loads this config as an extension."""
    if celer_config['options']['breathe']:
        from celerdirectives import CelerStructDirective
        app.add_directive('celerstruct', CelerStructDirective)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
]

if celer_config['options']['breathe']:
    extensions.append('breathe')
    breathe_default_project = 'celeritas'
    breathe_projects = {
        # See CMakeLists.txt
        breathe_default_project: build_dir / 'doxygen-xml'
    }
    breathe_default_members = ()
    breathe_show_include = False
    breathe_domain_by_extension = {
        "h": "cpp",
        "hh": "cpp",
        "cc": "cpp",
        "cu": "cpp",
    }

if celer_config['options']['sphinxbib']:
    import pybtex
    extensions.append("sphinxcontrib.bibtex")
    bibtex_bibfiles = [
        "_static/zotero.bib",
    ]
    bibtex_reference_style = 'author_year'

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
highlight_language = 'cpp'

# Enable numbered figures/tables
numfig = True

sys.path.insert(0, os.path.join(os.path.abspath('.'), "_python"))
import monkeysphinx

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'alabaster'
if celer_config['options']['furo']:
    html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/celeritas-square.svg"

if html_theme == 'alabaster':
    html_theme_options = {
        'github_button': True,
        'github_user': 'celeritas-project',
        'github_repo': project.lower(),
        'show_relbars': True,
        'show_powered_by': False,
    }

mathjax3_config = {
    # See _static/macros.tex
    "tex": {
        "macros": {
            "ee": r"\mathrm{e}",
            "dif": r"\;\mathrm{d}",
            "difd": [r"\frac{\mathrm{d}#1}{\mathrm{d}#2}", 2],
            "vd": r"\mathbf{\cdot}",
            "norm": [r"\|#1\|", 1],
            "abs": [r"|#1|", 1],
            "sgn": r"\mathop{\mathrm{sgn}}\nolimits",
        },
    },
    "loader": {"load": ["ui/lazy", "output/svg"]},
}

if int(sphinx_version.partition('.')[0]) < 4:
    # See https://mathjax.github.io/MathJax-demos-web/convert-configuration/convert-configuration.html
    mathjax_config = {"TeX": { "Macros": mathjax3_config["tex"]["macros"]}}

# -- Options for LaTeX output ------------------------------------------------


latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
'papersize': 'letterpaper',

'extraclassoptions': 'oneside',

# The font size ('10pt', '11pt' or '12pt').
'pointsize': '11pt',

# Additional stuff for the LaTeX preamble.
'preamble': r"""
\usepackage{sphinxcustom}
\usepackage{microtype}
\usepackage{pdfpages}
\usepackage{multirow}
\usepackage{fancyhdr} % Headers and footers
\usepackage{threeparttable}
\usepackage{tocloft}
\usepackage{etoolbox}
% Reset styles changed by sphinx.sty
\usepackage{ornltm-style}
\usepackage{ornltm-extract}
\input{./macros.tex}
""",

# Table of contents
'tableofcontents': r"""
\frontmatter
% Plain page
\thispagestyle{plain}%
\cleardoublepage
\tableofcontents
\listoftables
""",
# No chapter styles needed
'fncychap': "",
# Make references more robust to renumbering
'hyperref': r"""
\usepackage[hypertexnames=false]{hyperref}
\usepackage{hypcap}
\urlstyle{same}
""",
# Replace maketitle with generated title page:
# see http://texdoc.net/texmf-dist/doc/latex/pdfpages/pdfpages.pdf
# and documents repo:
 'maketitle': r"\includepdf[pages=-]{ornltm-header-celeritas.pdf}",
 'atendofbody': r"\includepdf[pages=-]{ornltm-footer.pdf}",
 # NOTE: '\titleclass{\section}{top}' breaks \printindex with a message about
 # "sphinxtheappendix" not being defined... since we don't care about the index
 # anyway, suppress it.
 'makeindex': "",
 'printindex': "",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'Celeritas.tex', 'Celeritas User Manual',
     author, 'howto'),
]

_latex_suffixes = frozenset(('.tex', '.sty', '.cls', '.pdf', '.png'))
latex_additional_files = [
    str(p) for p in Path('_static').iterdir()
    if p.suffix in _latex_suffixes
]
