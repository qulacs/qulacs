import textwrap

project = "C++/Python Project"

language = 'zh_TW'
locale_dirs = ['locale/']

# The `extensions` list should already be in here from `sphinx-quickstart`
extensions = [
    # there may be others here already, e.g. 'sphinx.ext.mathjax'
    'breathe',
    'exhale',
    'recommonmark'
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
    "rootFileTitle":         "應用程式接口",
    "doxygenStripFromPath":  "..",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    "treeViewIsBootstrap": True,
     "exhaleExecutesDoxygen": True,
     "exhaleDoxygenStdin": \
        "INPUT = ../../qulacs-src-min \n \
        OUTPUT_LANGUAGE = Chinese"

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