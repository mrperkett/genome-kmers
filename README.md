# Introduction

This Python package implements objects and functions that allow for fast and memory-efficient [*k*-mer](https://en.wikipedia.org/wiki/K-mer) calculations on the genome.  *k*-mer calculations are useful in a variety of areas in bioinformatics.  This implementation uses a data structure that is independent of the length of the *k-mer*.

It allows you to efficiently calculate:

- all unique k-mers in a genome and their frequency
- all unique k-mers shared between two or more genomes

Potential applications include:

- efficient first-pass at whole genome CRISPR guide design
    - identify all "good" CRISPR guides (i.e. guides that target a genome $< N$ times, $N$ usually equal to 1)
    - idenfity potential positive controls, which are predicted to kill cells via DNA damage response (i.e. guides that target a genome $>N$ times, usually $N > 5$)
    - identify all CRISPR guides that "cross-target" a collection of genomes (i.e. guides that target $< N$ times for each genome)
- efficient first-pass at primer design (i.e. identify all potential primers in the region of interest that target the genome $< N$ times as a first filter on primer design)
- using to build a suffix trie (though more memory efficient algorithms for this task exist)


**NOTE:** This is a alpha release with only `SequenceCollection` having been implemented.  `Kmers` is in dev and will be released soon.


# Setup

```shell
python3 -m pip install genome-kmers
```

## Basic usage

### SequenceCollection
`SequenceCollection` objects store a collection of sequence records into a single sequence byte array, which enables efficient downstream `Kmer` class calculations.  This class is optimized for *k-mer* calculations and is not meant to be a replacement for all the types of sequence manipulation that can be done.  You can initialize a `SequenceCollection` either by providing a fasta file path or a list of `(record_id, seq)` tuples.

To load using a list, using the keyword `sequence_list`.

```python
>>> from genome_kmers.sequence_collection import SequenceCollection
>>> seq_list = [("chr1", "ATCGAATTAG"), ("chr2", "GGATCTTGCATT"), ("chr3", "GTGATTGACCCCT")]
>>> seq_coll = SequenceCollection(sequence_list=seq_list)
```

By default, this will only load the forward strand into memory, which is what is typically desired for use with the `Kmers` class.  For certain applications, it may make sense to load either the `reverse_complement` or `both` into memory.  You can specify which strand(s) to load into memory using the keyword `strands_to_load`.

```python
>>> seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="both")
```

For most applications, you will want to initialize the SequenceCollection using a fasta file, such as can be downloaded from [NCBI](https://www.ncbi.nlm.nih.gov/guide/howto/dwn-genome/) or [Ensembl](https://useast.ensembl.org/info/data/ftp/index.html).  To initialize with a fasta file, use the keyword `fasta_file_path`.

```python
>>> seq_coll = SequenceCollection(fasta_file_path="example.fa")
```

Note that it is not allowed to provide both the `sequence_list` and `fasta_file_path` keywords, which will raise an exception.

```python
>>> seq_coll = SequenceCollection(sequence_list=seq_list, fasta_file_path="example.fa")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/mperkett/projects/kmer-counting/genome-kmers/src/genome_kmers/sequence_collection.py", line 129, in __init__
    raise ValueError(
ValueError: Either fasta_file_path or sequence_list must be specified.  Bothcannot be specified.
```

Once you have loaded a `SequenceCollection`, you can get the corresponding fasta represntation using the `str` class method.

```python
>>> print(str(seq_coll))
>chr1
ATCGAATTAG
>chr2
GGATCTTGCATT
>chr3
GTGATTGACCCCT
```

If you `reverse_complement` the SequenceCollection, this internally reverse complements the sequence byte array representation and printing `seq_coll` will give reverse complemented sequences.  Note that the record order remains the same (i.e. "chr1" is still printed first in this example).

```python
>>> seq_coll.reverse_complement()
>>> print(str(seq_coll))
>chr1
CTAATTCGAT
>chr2
AATGCAAGATCC
>chr3
AGGGGTCAATCAC
```

Note that `reverse_complement` is undefined if both strands have been loaded and will raise the following exception.

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/mperkett/projects/kmer-counting/genome-kmers/src/genome_kmers/sequence_collection.py", line 682, in reverse_complement
    raise ValueError(f"self._strands_loaded ({self._strands_loaded}) cannot be 'both'")
ValueError: self._strands_loaded (both) cannot be 'both'
```

You can also iterate over SequenceCollection records using `iter_records`.  This method yields the record name along with the start and end indices of the sequence as stored in the sequence byte array.  This is primarily used for downstream `Kmer` class calculations.

```python
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
```

The `Kmer` class defines a *k-mer* by its `SequenceCollection` byte array index.  As such, it is often required to determine with which sequence record a $k-mer$ is associated from only the sequence byte array index.  This can be determined in varying levels of detail using `get_record_loc_from_sba_index`, `get_record_name_from_sba_index`, and `get_segment_num_from_sba_index`.

```python
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
```

**Note**, as you can see from above, the sequence index returned is 0-based.  Convention within the field is to report sequences as 1-based indices.  The decision to use 0-based indices was made to simplify the `Kmer` class implementation.

### Kmers

**NOTE:** `SequenceCollection` has been implemented, but `Kmers` is in dev and will be released soon.


# Development

## Setup
### Using [poetry](https://python-poetry.org/) for environment and dependency management

```bash
git clone git@github.com:mrperkett/genome-kmers.git
cd genome-kmers/

poetry install

# run tests to verify everything is working properly
poetry run python -m pytest
```

## Profiling

You can choose between running the small (default), medium, or large test sets with the appropriate command line flag.  It takes 8 seconds to run the small, 25 seconds for the medium, and 8 mins for the large.  The full set of options are given below.

### Example commands

```bash
# activate the dev venv (either `poetry shell` or `pyenv <venv_name>`)

# run all (small parameter set)
python3 profiling/run_profiling.py

# run single category of profiling
python3 profiling/run_profiling.py -c fasta_init

# run a different number of iterations
python3 profiling/run_profiling.py -n 10 -c fasta_init

# save profiling info to file
python3 profiling/run_profiling.py -s small -c fasta_init -o output/profiling-small

# run all profiling (large parameter set)
python3 profiling/run_profiling.py -s large -o output/profiling-large
```

### Call signature

```shell
$ python3 profiling/run_profiling.py --help
usage: run_profiling.py [-h] [--run-size {small,medium,large}] [--num-iter NUM_ITER]
                        [--category {all,seq_list_init,fasta_init,get_segment_num}] [--output-base OUTPUT_BASE]

options:
  -h, --help            show this help message and exit
  --run-size {small,medium,large}, -s {small,medium,large}
                        size of the profiling run. 'small' is quick, but incomplete. 'large' is slower, but more
                        complete.
  --num-iter NUM_ITER, -n NUM_ITER
                        number of run times over which to average. Must be > 0.
  --category {all,seq_list_init,fasta_init,get_segment_num}, -c {all,seq_list_init,fasta_init,get_segment_num}
  --output-base OUTPUT_BASE, -o OUTPUT_BASE
                        output file base that is used as the prefix for profiling output files
```



### Output v0.1.0
#### run all profiling (small)
```shell
$ poetry shell
$ python3 profiling/run_profiling.py

INFO:root:sequence list init: 'forward' strand
INFO:root:
   total_seq_len  run_time_0  run_time_1  run_time_2  avg_run_time
0         1000.0    0.001331    0.001158    0.001150      0.001213
1        10000.0    0.001242    0.001267    0.001262      0.001257
2       100000.0    0.002600    0.002578    0.002484      0.002554
INFO:root:sequence list init: 'reverse_complement' strand
INFO:root:
   total_seq_len  run_time_0  run_time_1  run_time_2  avg_run_time
0         1000.0    0.001276    0.001170    0.001131      0.001192
1        10000.0    0.001319    0.001255    0.001273      0.001282
2       100000.0    0.002586    0.002523    0.002485      0.002532
INFO:root:sequence list init: 'both' strand
INFO:root:
   total_seq_len  run_time_0  run_time_1  run_time_2  avg_run_time
0         1000.0    0.001184    0.001187    0.001124      0.001165
1        10000.0    0.001260    0.001259    0.001252      0.001257
2       100000.0    0.002589    0.002497    0.002514      0.002533
INFO:root:Total init profiling run time: 2.7478222846984863
INFO:root:get_segment_num_from_sba_index: 'forward' strand
INFO:root:
   total_seq_len  num_chromosomes  num_lookups  run_time_0  run_time_1  run_time_2  avg_run_time
0      1000000.0              1.0      10000.0    0.018413    0.018719    0.018563      0.018565
1      1000000.0             10.0      10000.0    0.019644    0.019572    0.019537      0.019584
2      1000000.0            100.0      10000.0    0.020824    0.020844    0.020935      0.020868
INFO:root:get_segment_num_from_sba_index: 'reverse_complement' strand
INFO:root:
   total_seq_len  num_chromosomes  num_lookups  run_time_0  run_time_1  run_time_2  avg_run_time
0      1000000.0              1.0      10000.0    0.018890    0.018635    0.018785      0.018770
1      1000000.0             10.0      10000.0    0.020107    0.020049    0.020038      0.020065
2      1000000.0            100.0      10000.0    0.021526    0.021306    0.021351      0.021394
INFO:root:get_segment_num_from_sba_index: 'both' strand
INFO:root:
   total_seq_len  num_chromosomes  num_lookups  run_time_0  run_time_1  run_time_2  avg_run_time
0      1000000.0              1.0      10000.0    0.019012    0.019060    0.018949      0.019007
1      1000000.0             10.0      10000.0    0.019918    0.021051    0.020003      0.020324
2      1000000.0            100.0      10000.0    0.021205    0.021208    0.021269      0.021228
INFO:root:Total get_segment_num_from_sba_index run time: 4.917483568191528
INFO:root:profile_fasta_init
INFO:root:
   total_seq_len  num_chromosomes  max_line_length  run_time_0  run_time_1  run_time_2  avg_run_time
0           1000               10               80    0.001245    0.001212    0.001158      0.001205
1          10000               10               80    0.001377    0.001410    0.001371      0.001386
2         100000               10               80    0.003517    0.003403    0.003380      0.003433
INFO:root:Total fasta_init_profiling run time: 0.06989192962646484
```

#### run single category of profiling
```shell
$ poetry shell
$ python3 profiling/run_profiling.py -c fasta_init

INFO:root:profile_fasta_init
INFO:root:
   total_seq_len  num_chromosomes  max_line_length  run_time_0  run_time_1  run_time_2  avg_run_time
0           1000               10               80    0.001286    0.001277    0.001226      0.001263
1          10000               10               80    0.001377    0.001436    0.001370      0.001394
2         100000               10               80    0.003570    0.003672    0.003399      0.003547
INFO:root:Total fasta_init_profiling run time: 2.4503276348114014
```

#### run a different number of iterations
```shell
$ poetry shell
$ python3 profiling/run_profiling.py -n 10 -c fasta_init

INFO:root:profile_fasta_init
INFO:root:
   total_seq_len  num_chromosomes  max_line_length  run_time_0  ...  run_time_7  run_time_8  run_time_9  avg_run_time
0           1000               10               80    0.001279  ...    0.001176    0.001134    0.001132      0.001192
1          10000               10               80    0.001345  ...    0.001412    0.001479    0.001360      0.001391
2         100000               10               80    0.003591  ...    0.003442    0.003517    0.003689      0.003519

[3 rows x 14 columns]
INFO:root:Total fasta_init_profiling run time: 2.6365883350372314
```

#### save profiling info to file
```shell
$ poetry shell
$ python3 profiling/run_profiling.py -s small -c fasta_init -o output/profiling-small

INFO:root:profile_fasta_init
INFO:root:
   total_seq_len  num_chromosomes  max_line_length  run_time_0  run_time_1  run_time_2  avg_run_time
0           1000               10               80    0.001352    0.001320    0.001313      0.001328
1          10000               10               80    0.001366    0.001366    0.001415      0.001382
2         100000               10               80    0.003696    0.003578    0.003543      0.003606
INFO:root:profiling info written to 'output/profiling-small-fasta-init.csv'
INFO:root:Total fasta_init_profiling run time: 2.4825077056884766
```

#### run all profiling (large)
```shell
$ poetry shell
$ python3 profiling/run_profiling.py -s large -o output/profiling-large

INFO:root:sequence list init: 'forward' strand
INFO:root:
   total_seq_len  run_time_0  run_time_1  run_time_2  avg_run_time
0        10000.0    0.001328    0.001242    0.001301      0.001290
1       100000.0    0.002419    0.002465    0.002397      0.002427
2      1000000.0    0.015227    0.015810    0.015799      0.015612
3     10000000.0    0.156373    0.148769    0.156308      0.153817
4    100000000.0    1.545371    1.527991    1.489191      1.520851
INFO:root:profiling info written to 'output/profiling-large-seq-list-init-forward.csv'
INFO:root:sequence list init: 'reverse_complement' strand
INFO:root:
   total_seq_len  run_time_0  run_time_1  run_time_2  avg_run_time
0        10000.0    0.001451    0.001270    0.001240      0.001320
1       100000.0    0.002521    0.002613    0.002595      0.002577
2      1000000.0    0.014952    0.015979    0.015842      0.015591
3     10000000.0    0.158836    0.153975    0.157310      0.156707
4    100000000.0    1.592198    1.605245    1.553978      1.583807
INFO:root:profiling info written to 'output/profiling-large-seq-list-init-reverse_complement.csv'
INFO:root:sequence list init: 'both' strand
INFO:root:
   total_seq_len  run_time_0  run_time_1  run_time_2  avg_run_time
0        10000.0    0.001362    0.001338    0.001216      0.001305
1       100000.0    0.002667    0.002818    0.002800      0.002762
2      1000000.0    0.015171    0.016268    0.016207      0.015882
3     10000000.0    0.160919    0.150851    0.162474      0.158081
4    100000000.0    1.605487    1.575450    1.531359      1.570765
INFO:root:profiling info written to 'output/profiling-large-seq-list-init-both.csv'
INFO:root:Total init profiling run time: 181.93565201759338
INFO:root:get_segment_num_from_sba_index: 'forward' strand
INFO:root:
   total_seq_len  num_chromosomes  num_lookups  run_time_0  run_time_1  run_time_2  avg_run_time
0      1000000.0              1.0      10000.0    0.018799    0.018474    0.018674      0.018649
1      1000000.0             10.0      10000.0    0.019941    0.020293    0.020060      0.020098
2      1000000.0            100.0      10000.0    0.020934    0.020939    0.039381      0.027085
3      1000000.0           1000.0      10000.0    0.022121    0.022188    0.022214      0.022174
4      1000000.0          10000.0      10000.0    0.023363    0.023281    0.023288      0.023310
5      1000000.0         100000.0      10000.0    0.024577    0.024755    0.024687      0.024673
6      1000000.0        1000000.0      10000.0    0.025940    0.025862    0.025810      0.025871
INFO:root:profiling info written to 'output/profiling-large-segment-num-from-sba-index-forward.csv'
INFO:root:get_segment_num_from_sba_index: 'reverse_complement' strand
INFO:root:
   total_seq_len  num_chromosomes  num_lookups  run_time_0  run_time_1  run_time_2  avg_run_time
0      1000000.0              1.0      10000.0    0.018739    0.018864    0.018157      0.018587
1      1000000.0             10.0      10000.0    0.017886    0.018129    0.019687      0.018567
2      1000000.0            100.0      10000.0    0.021143    0.020960    0.021166      0.021090
3      1000000.0           1000.0      10000.0    0.022239    0.022299    0.022220      0.022252
4      1000000.0          10000.0      10000.0    0.023479    0.023767    0.023435      0.023561
5      1000000.0         100000.0      10000.0    0.024818    0.024772    0.024850      0.024813
6      1000000.0        1000000.0      10000.0    0.026086    0.025692    0.025994      0.025924
INFO:root:profiling info written to 'output/profiling-large-segment-num-from-sba-index-reverse_complement.csv'
INFO:root:get_segment_num_from_sba_index: 'both' strand
INFO:root:
   total_seq_len  num_chromosomes  num_lookups  run_time_0  run_time_1  run_time_2  avg_run_time
0      1000000.0              1.0      10000.0    0.018937    0.018914    0.019014      0.018955
1      1000000.0             10.0      10000.0    0.019967    0.020011    0.019970      0.019983
2      1000000.0            100.0      10000.0    0.021246    0.021211    0.021278      0.021245
3      1000000.0           1000.0      10000.0    0.022763    0.022495    0.022814      0.022691
4      1000000.0          10000.0      10000.0    0.023838    0.023717    0.023832      0.023796
5      1000000.0         100000.0      10000.0    0.024851    0.024572    0.024801      0.024741
6      1000000.0        1000000.0      10000.0    0.025449    0.025377    0.025465      0.025431
INFO:root:profiling info written to 'output/profiling-large-segment-num-from-sba-index-both.csv'
INFO:root:Total get_segment_num_from_sba_index run time: 134.99651336669922
INFO:root:profile_fasta_init
INFO:root:
   total_seq_len  num_chromosomes  max_line_length  run_time_0  run_time_1  run_time_2  avg_run_time
0          10000               10               80    0.001474    0.001406    0.001453      0.001444
1         100000               10               80    0.003463    0.003411    0.003384      0.003419
2        1000000               10               80    0.023944    0.025002    0.024875      0.024607
3       10000000               10               80    0.244484    0.235779    0.240098      0.240121
4      100000000               10               80    2.472731    2.473906    2.404620      2.450419
INFO:root:profiling info written to 'output/profiling-large-fasta-init.csv'
INFO:root:Total fasta_init_profiling run time: 65.84908413887024
```