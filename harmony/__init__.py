from .harmony import harmonize

from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version('harmony-pytorch')
    del version
except PackageNotFoundError:
    pass
