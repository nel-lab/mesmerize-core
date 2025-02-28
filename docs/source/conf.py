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
from bs4 import BeautifulSoup
from typing import *
import mesmerize_core

# -- Project information -----------------------------------------------------

project = "mesmerize-core"
copyright = "2023, Kushal Kolar, Caitlin Lewis, Arjun Putcha"
author = "Kushal Kolar, Caitlin Lewis, Arjun Putcha"

# The full version, including alpha/beta/rc tags
release = mesmerize_core.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon"]
autodoc_typehints = "description"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {"page_sidebar_items": ["class_page_toc"]}
autoclass_content = "both"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autodoc_member_order = "bysource"


def _setup_navbar_side_toctree(app: Any):
    def add_class_toctree_function(
        app: Any, pagename: Any, templatename: Any, context: Any, doctree: Any
    ):
        def get_class_toc() -> Any:
            soup = BeautifulSoup(context["body"], "html.parser")

            matches = soup.find_all("dl")
            if matches is None or len(matches) == 0:
                return ""
            items = []
            deeper_depth = matches[0].find("dt").get("id").count(".")
            for match in matches:
                match_dt = match.find("dt")
                if match_dt is not None and match_dt.get("id") is not None:
                    current_title = match_dt.get("id")
                    current_depth = match_dt.get("id").count(".")
                    current_link = match.find(class_="headerlink")
                    if current_link is not None:
                        if deeper_depth > current_depth:
                            deeper_depth = current_depth
                        if deeper_depth == current_depth:
                            items.append(
                                {
                                    "title": current_title.split(".")[-1],
                                    "link": current_link["href"],
                                    "attributes_and_methods": [],
                                }
                            )
                        if deeper_depth < current_depth:
                            items[-1]["attributes_and_methods"].append(
                                {
                                    "title": current_title.split(".")[-1],
                                    "link": current_link["href"],
                                }
                            )
            return items

        context["get_class_toc"] = get_class_toc

    app.connect("html-page-context", add_class_toctree_function)


def setup(app: Any):
    for setup_function in [
        _setup_navbar_side_toctree,
    ]:
        setup_function(app)
