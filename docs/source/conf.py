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
import os
import sys

# Separate pages for each module
# sphinx-apidoc -o .\source .. -f --separate
# xcopy .\_build\html\ .\ /s /y /d

# sys.path.insert(0, os.path.abspath("..."))
sys.path.insert(0, os.path.abspath("../../"))
print("path ...: ", os.path.abspath("..."))
print("path ../../src: ", os.path.abspath("../../src"))
print("path ../../src/fulcheranalyzer: ", os.path.abspath("../../src/fulcheranalyzer"))
print("path ../../: ", os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "Fulcher Analyzer"
copyright = "2022, A.K."
author = "A.K."


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "jupyter_sphinx",
    "sphinx.ext.napoleon",
    "myst_nb",  # conda install -c conda-forge myst-nb
    "sphinx_copybutton",  # conda install -c conda-forge sphinx-copybutton
    "sphinxcontrib.video",  # pip install sphinxcontrib-video
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# shpinx.ext.napoleon settings
# napoleon_google_docstring = False

# The master toctree document.
master_doc = "index"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
# html_theme = "python_docs_theme"
# html_theme = "renku" # pip install renku-sphinx-theme
html_theme = "pydata_sphinx_theme"  # conda install pydata-sphinx-theme --channel conda-forge

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/h2icon.png"

html_theme_options = {
    "logo": {"text": "Fulcher Analyzer"},
    "pygment_light_style": "tango",
    "pygment_dark_style": "native",
    "favicons": [{"rel": "icon", "sizes": "32x32", "href": "h2icon.png",},],
}

