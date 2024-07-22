Development
###########


Setup
=====

The following instructions use `poetry <https://python-poetry.org/>`_ for environment and dependency management.

.. code-block:: bash

    git clone git@github.com:mrperkett/genome-kmers.git
    cd genome-kmers/

    poetry install

    # run tests to verify everything is working properly
    poetry run python -m pytest

Profiling
=========

You can choose between running the small (default), medium, or large test sets with the appropriate command line flag.  It takes 8 seconds to run the small, 25 seconds for the medium, and 8 mins for the large.  The full set of options are given below.

Example commands
----------------

.. code-block:: bash

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

Call signature
--------------

.. code-block:: bash

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


Output v0.1.0
-------------

run all profiling (small)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

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

run single category of profiling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ poetry shell
    $ python3 profiling/run_profiling.py -c fasta_init

    INFO:root:profile_fasta_init
    INFO:root:
    total_seq_len  num_chromosomes  max_line_length  run_time_0  run_time_1  run_time_2  avg_run_time
    0           1000               10               80    0.001286    0.001277    0.001226      0.001263
    1          10000               10               80    0.001377    0.001436    0.001370      0.001394
    2         100000               10               80    0.003570    0.003672    0.003399      0.003547
    INFO:root:Total fasta_init_profiling run time: 2.4503276348114014

run a different number of iterations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

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

save profiling info to file
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

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

run all profiling (large)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

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