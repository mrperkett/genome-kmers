.. genome-kmers documentation master file, created by
   sphinx-quickstart on Thu Jun 13 15:42:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to genome-kmers's documentation!
========================================

This project contains a collection of classes useful for generating all k-mers in the genome in a memory-efficient manner.  The amount of memory required scales with the genome length, but does not depend on the length of the k-mer.  K-mers are stored as unsigned 32 bit integers that reference their starting location in the genome.  This package makes it simple to load all the k-mers for a fasta-formatted genome file and perform various k-mer calculations including counting the number of unique k-mers.  This package uses `numba <https://numba.pydata.org/>`_ to perform efficient k-mer operations and is orders of magnitude faster than pure Python.

.. toctree::
    :maxdepth: 2

    Overview <overview>
    API Reference <genome_kmers>
    Examples <examples>
    Development <development>

* :ref:`genindex`
