import textwrap

project = "Qulacs"

language = 'ja'
locale_dirs = ['locale/']

# The `extensions` list should already be in here from `sphinx-quickstart`
extensions = [
    # there may be others here already, e.g. 'sphinx.ext.mathjax'
    # there may be others here already, e.g. 'sphinx.ext.mathjax'
    'breathe',
    'exhale',
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
]

exclude_patterns = ['_build', '**.ipynb_checkpoints']
nbsphinx_allow_errors = True
myst_enable_extensions = [
    "dollarmath",
]


# source files for shpinx
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# Setup the breathe extension
breathe_projects = {
    "Docs": "./xml"
}

breathe_default_project = "Docs"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./api",
    "rootFileName":          "cpp_library_root.rst",
    "rootFileTitle":         "C++ APIリファレンス",
    "doxygenStripFromPath":  "..",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": \
        "INPUT = ../../../src/cppsim ../../../src/vqcsim \n \
        OUTPUT_LANGUAGE = Japanese-en \n \
        FILE_PATTERNS          = *.hpp \n \
        WARN_IF_UNDOCUMENTED   = NO \n \
        ENABLE_PREPROCESSING   = YES \n \
        MACRO_EXPANSION        = YES \n \
        PREDEFINED             += __attribute__(x)= \n \
        PREDEFINED             += DllExport= \n \
        GENERATE_LEGEND        = YES \n \
        GRAPHICAL_HIERARCHY    = YES \n \
        CLASS_GRAPH            = YES \n \
        HIDE_UNDOC_RELATIONS   = YES\n \
        CLASS_DIAGRAMS         = YES\n"
}

# Tell sphinx what the primary language being documented is.
#primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
#highlight_language = 'cpp'

#

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
import os

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


html_theme_options = {
    # 'canonical_url': '',
    # 'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': False,
    # 'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # 'vcs_pageview_mode': '',
    'style_nav_header_background': '#004659',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 2,
    'includehidden': True,
    'titles_only': False
}

templates_path = ["_templates"]
html_static_path = ['_static']

copyright = '2018 Qulacs Authors'
version = '0.5.4'
