Overview
########

This Python package implements objects and functions that allow for fast and memory-efficient `k-mer <https://en.wikipedia.org/wiki/K-mer>`_ calculations on the genome.  *k*-mer calculations are useful in a variety of areas in bioinformatics.  This implementation uses a data structure that is independent of the length of the *k-mer*.

It allows you to efficiently calculate:

* all unique k-mers in a genome and their frequency
* all unique k-mers shared between two or more genomes

Potential applications include:

* efficient first-pass at whole genome CRISPR guide design

    * identify all "good" CRISPR guides (i.e. guides that target a genome < N times, N usually equal to 1)
    * idenfity potential positive controls, which are predicted to kill cells via DNA damage response (i.e. guides that target a genome >N times, usually N > 5)
    * identify all CRISPR guides that "cross-target" a collection of genomes (i.e. guides that target < N times for each genome)

* efficient first-pass at primer design (i.e. identify all potential primers in the region of interest that target the genome < N times as a first filter on primer design)
* using to build a suffix trie (though more memory efficient algorithms for this task exist)


**NOTE:** This is a alpha release with only ``SequenceCollection`` having been implemented.  ``Kmers`` is in dev and will be released soon.


Setup
=====

.. code-block:: bash

    python3 -m pip install genome-kmers



Basic usage
===========

SequenceCollection
------------------

``SequenceCollection`` objects store a collection of sequence records into a single sequence byte array, which enables efficient downstream ``Kmer`` class calculations.  This class is optimized for *k-mer* calculations and is not meant to be a replacement for all the types of sequence manipulation that can be done.  You can initialize a ``SequenceCollection`` either by providing a fasta file path or a list of ``(record_id, seq)`` tuples.

To load using a list, using the keyword ``sequence_list``.

.. code-block:: python

    >>> from genome_kmers.sequence_collection import SequenceCollection
    >>> seq_list = [("chr1", "ATCGAATTAG"), ("chr2", "GGATCTTGCATT"), ("chr3", "GTGATTGACCCCT")]
    >>> seq_coll = SequenceCollection(sequence_list=seq_list)


By default, this will only load the forward strand into memory, which is what is typically desired for use with the ``Kmers`` class.  For certain applications, it may make sense to load either the ``reverse_complement`` or ``both`` into memory.  You can specify which strand(s) to load into memory using the keyword ``strands_to_load``.

.. code-block:: python

    >>> seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="both")

For most applications, you will want to initialize the SequenceCollection using a fasta file, such as can be downloaded from `NCBI <https://www.ncbi.nlm.nih.gov/guide/howto/dwn-genome/>`_ or `Ensembl <https://useast.ensembl.org/info/data/ftp/index.html>`.  To initialize with a fasta file, use the keyword ``fasta_file_path``.

.. code-block:: python

    >>> seq_coll = SequenceCollection(fasta_file_path="example.fa")

Note that it is not allowed to provide both the ``sequence_list`` and ``fasta_file_path`` keywords, which will raise an exception.

.. code-block:: python

    >>> seq_coll = SequenceCollection(sequence_list=seq_list, fasta_file_path="example.fa")
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/home/mperkett/projects/kmer-counting/genome-kmers/src/genome_kmers/sequence_collection.py", line 129, in __init__
        raise ValueError(
    ValueError: Either fasta_file_path or sequence_list must be specified.  Bothcannot be specified.

Once you have loaded a ``SequenceCollection``, you can get the corresponding fasta represntation using the ``str`` class method.

.. code-block:: python

    >>> print(str(seq_coll))
    >chr1
    ATCGAATTAG
    >chr2
    GGATCTTGCATT
    >chr3
    GTGATTGACCCCT

If you ``reverse_complement`` the SequenceCollection, this internally reverse complements the sequence byte array representation and printing ``seq_coll`` will give reverse complemented sequences.  Note that the record order remains the same (i.e. "chr1" is still printed first in this example).

.. code-block:: python

    >>> seq_coll.reverse_complement()
    >>> print(str(seq_coll))
    >chr1
    CTAATTCGAT
    >chr2
    AATGCAAGATCC
    >chr3
    AGGGGTCAATCAC

Note that ``reverse_complement`` is undefined if both strands have been loaded and will raise the following exception.

.. code-block::

    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/home/mperkett/projects/kmer-counting/genome-kmers/src/genome_kmers/sequence_collection.py", line 682, in reverse_complement
        raise ValueError(f"self._strands_loaded ({self._strands_loaded}) cannot be 'both'")
    ValueError: self._strands_loaded (both) cannot be 'both'

You can also iterate over SequenceCollection records using ``iter_records``.  This method yields the record name along with the start and end indices of the sequence as stored in the sequence byte array.  This is primarily used for downstream ``Kmer`` class calculations.

.. code-block:: python

    >>> seq_coll = SequenceCollection(sequence_list=seq_list)
    >>> for record_name, sba_seg_start_idx, sba_seg_end_idx in seq_coll.iter_records():
    ...    print(f"{record_name}")
    ...    print(f"\tseq byte array start index: {sba_seg_start_idx}")
    ...    print(f"\tseq byte array end index: {sba_seg_end_idx}")

    chr1
            seq byte array start index: 0
            seq byte array end index: 9
    chr2
            seq byte array start index: 11
            seq byte array end index: 22
    chr3
            seq byte array start index: 24
            seq byte array end index: 36

The ``Kmer`` class defines a *k-mer* by its ``SequenceCollection`` byte array index.  As such, it is often required to determine with which sequence record a $k-mer$ is associated from only the sequence byte array index.  This can be determined in varying levels of detail using ``get_record_loc_from_sba_index``, ``get_record_name_from_sba_index``, and ``get_segment_num_from_sba_index``.

.. code-block:: python

    >>> # chr1: index = 5
    >>> strand, record_name, seq_idx = seq_coll.get_record_loc_from_sba_index(5)
    >>> print(f"{strand}{record_name}:{seq_idx}")
    +chr1:5
    >>> # chr2, index = 0
    >>> strand, record_name, seq_idx = seq_coll.get_record_loc_from_sba_index(11)
    >>> print(f"{strand}{record_name}:{seq_idx}")
    +chr2:0
    >>> # chr3, index = 2
    >>> strand, record_name, seq_idx = seq_coll.get_record_loc_from_sba_index(26)
    >>> print(f"{strand}{record_name}:{seq_idx}")
    +chr3:2

**Note**, as you can see from above, the sequence index returned is 0-based.  Convention within the field is to report sequences as 1-based indices.  The decision to use 0-based indices was made to simplify the ``Kmer`` class implementation.

Kmers
-----

**NOTE:** ``SequenceCollection`` has been implemented, but ``Kmers`` is in dev and will be released soon.
