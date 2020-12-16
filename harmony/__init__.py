from .harmony import harmonize

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # < Python 3.8: Use backport module
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version('harmony-pytorch')
    del version
except PackageNotFoundError:
    pass
