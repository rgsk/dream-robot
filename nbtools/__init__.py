# notebook helpers, importable from any folder (the editable install puts the repo root on
# sys.path):
#     from nbtools import NotebookCache, probe, show
from nbtools.notebook_cache import NotebookCache
from nbtools.probe import probe, show

__all__ = ['NotebookCache', 'probe', 'show']
