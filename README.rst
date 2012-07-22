==================================================
simserver -- document similarity server in Python
==================================================


Index plain text documents and query the index for semantically related documents.

Simserver uses transactions internally to provide a robust and scalable similarity server.


Installation
------------

Simserver builds on the `gensim <http://radimrehurek.com/gensim/>`_ framework for
topic modelling.

The simple way to install `simserver` is with::

    sudo easy_install -U simserver

Or, if you have instead downloaded and unzipped the `source tar.gz <http://pypi.python.org/pypi/simserver>`_ package,
you'll need to run::

    python setup.py test
    sudo python setup.py install

This version has been tested under Python 2.5 and 2.7, but should run on any 2.5 <= Python < 3.0.

Documentation
-------------

See http://radimrehurek.com/gensim/simserver.html . More coming soon.

Licensing
----------------

Simserver is released under the `GNU Affero GPL license v3 <http://www.gnu.org/licenses/agpl.html>`_.

This means you may use simserver freely in your application (even commercial application!),
but you **must then open-source your application as well**, under an AGPL-compatible license.

The AGPL license makes sure that this applies even when you make your application
available only remotely (such as through the web).

TL;DR: **simserver is open-source, but you have to contact me for any proprietary use.**

History
-------------

0.1.4:
  * performance improvements to sharding
  * change to threading model -- removed restriction on per-thread session access
  * bug fix in index optimize()

0.1.3: 
  * changed behaviour for very few training documents: instead of latent semantic analysis, use simpler log-entropy model
  * fixed bug with leaking SQLite file descriptors

-------------

Copyright (c) 2009-2012 Radim Rehurek
