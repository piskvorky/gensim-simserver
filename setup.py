#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU AGPL v3.0 - http://www.gnu.org/licenses/agpl.html

"""
Run with:

sudo python ./setup.py install
"""

import os
import sys

if sys.version_info[:2] < (2, 5):
    raise Exception('This version of simserver needs Python 2.5 or later. ')

import ez_setup
ez_setup.use_setuptools()
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()



setup(
    name = 'simserver',
    version = '0.1.4',
    description = 'Document similarity server',
    long_description = read('README.rst'),

    packages = find_packages(),

    # there is a bug in python2.5, preventing distutils from using any non-ascii characters :( http://bugs.python.org/issue2562
    author = 'Radim Rehurek', # u'Radim Řehůřek', # <- should really be this...
    author_email = 'radimrehurek@seznam.cz',

    url = 'https://github.com/piskvorky/gensim-simserver',
    download_url = 'http://pypi.python.org/pypi/simserver',

    keywords = 'Similarity server, document database, Latent Semantic Indexing, LSA, '
    'LSI, LDA, Latent Dirichlet Allocation, TF-IDF, gensim',

    license = 'AGPL v3',
    platforms = 'any',

    zip_safe = False,

    classifiers = [ # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],

    test_suite = "simserver.test",

    install_requires = [
        'gensim >= 0.8.5',
        'Pyro4 >= 4.8',
        'sqlitedict >= 1.0.8',
    ],

    include_package_data = True,

)
