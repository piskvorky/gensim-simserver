"""
Package containing a document similarity server, an extension of gensim.
"""

# for IPython tab-completion
from simserver import SessionServer, SimServer


try:
    __version__ = __import__('pkg_resources').get_distribution('simserver').version
except:
    __version__ = '?'


import logging

class NullHandler(logging.Handler):
    """For python versions <= 2.6; same as `logging.NullHandler` in 2.7."""
    def emit(self, record):
        pass

logger = logging.getLogger('simserver')
if len(logger.handlers) == 0:	# To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())
