# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "RecSys Examples"
copyright = "Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved."
author = "Nvidia"
release = "v0.1.0"
html_show_sphinx = False

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinxarg.ext",
    "sphinx_click",
    "sphinx_copybutton",
    "myst_parser",
]

myst_heading_anchors = 4

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

autosummary_generate = True

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_typehints = "none"
napoleon_google_docstring = True
napoleon_numpy_docstring = False


def remove_class_signature(app, what, name, obj, options, signature, return_annotation):
    # remove args list from signature
    if what == "class":
        return ("", return_annotation)
    return (signature, return_annotation)


def setup(app):
    app.connect("autodoc-process-signature", remove_class_signature)
