# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import datetime
import os
import json


# -- Project information -----------------------------------------------------

project = 'Celeritas'
all_authors = [
 'Seth R Johnson',
 # Remaining authors in alphabetical order
 'Tom Evans',
 'Soon Yung Jun',
 'Guilherme Lima',
 'Amanda Lund',
 'Ben Morgan',
 'Vince Pascuzzi',
 'Stefano C Tognini',
]
author = " and ".join(all_authors)
copyright = '{:%Y}, UT–Battelle/ORNL'.format(datetime.datetime.today())

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
try:
    build_dir = os.environ['CMAKE_CURRENT_BINARY_DIR']
    with open(os.path.join(build_dir, 'config.json'), 'r') as f:
        celer_config = json.load(f)
except (KeyError, IOError) as e:
    print("Failed to load config data:", e)
    build_dir = '.'
    celer_config = {
        "version": "unknown",
        "release": "unknown",
        "breathe": False,
        "sphinxbib": False,
    }

version = celer_config['version']
release = celer_config['release']

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.todo'
]

if celer_config['breathe']:
    extensions.append('breathe')
    breathe_default_project = project
    breathe_projects = {
        project: os.path.join(build_dir, 'xml')
    }
    breathe_default_members = ('members',)

if celer_config['sphinxbib']:
    import pybtex
    extensions.append("sphinxcontrib.bibtex")
    bibtex_bibfiles = ['_static/references.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
highlight_language = 'cpp'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/celeritas-thumbnail.png"

