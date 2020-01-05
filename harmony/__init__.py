from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import logging
import sys

logger = logging.getLogger("harmony")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

from .harmony import harmonize
