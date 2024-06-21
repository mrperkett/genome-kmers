# Introduction

# Setup

```
poetry install
```

# Profiling

You can choose between running the small, medium (default), or large test sets with the appropriate command line flag.  It takes 8 seconds to run the small, 25 seconds for the medium, and 8 mins for the large.  The full set of options are given below.

```shell
$ python3 profiling/run_profiling.py --help
usage: run_profiling.py [-h] [--small | --medium | --large] [--num-iter NUM_ITER]

options:
  -h, --help            show this help message and exit
  --small               Run the small test set
  --medium              Run the medium test set
  --large               Run the full test set
  --num-iter NUM_ITER, -n NUM_ITER
                        number of run times over which to average. Must be > 0.
```



## v0.1.0
```shell
$ poetry shell
$ python3 profiling/run_profiling.py

INFO:root:sequence list init: 'forward' strand
INFO:root:
   total_seq_len  run_time_0  run_time_1  run_time_2  avg_run_time
0        10000.0    0.003764    0.003862    0.003760      0.003795
1       100000.0    0.004998    0.004932    0.005120      0.005017
2      1000000.0    0.017553    0.018794    0.018660      0.018336
3     10000000.0    0.159049    0.154475    0.158284      0.157270
4    100000000.0    1.593724    1.557280    1.513361      1.554788
INFO:root:sequence list init: 'reverse_complement' strand
INFO:root:
   total_seq_len  run_time_0  run_time_1  run_time_2  avg_run_time
0        10000.0    0.004003    0.003945    0.003890      0.003946
1       100000.0    0.005212    0.005291    0.005191      0.005231
2      1000000.0    0.018693    0.018770    0.018941      0.018801
3     10000000.0    0.157152    0.151918    0.159417      0.156163
4    100000000.0    1.634692    1.590210    1.522289      1.582397
INFO:root:sequence list init: 'both' strand
INFO:root:
   total_seq_len  run_time_0  run_time_1  run_time_2  avg_run_time
0        10000.0    0.003833    0.003736    0.005083      0.004217
1       100000.0    0.009911    0.006826    0.005335      0.007357
2      1000000.0    0.017900    0.018841    0.018697      0.018479
3     10000000.0    0.157986    0.153353    0.162349      0.157896
4    100000000.0    1.619174    1.601206    1.603060      1.607813
INFO:root:get_segment_num_from_sba_index: 'forward' strand
INFO:root:
   total_seq_len  num_chromosomes  num_lookups  run_time_0  run_time_1  run_time_2  avg_run_time
0      1000000.0              1.0      10000.0    0.037178    0.038602    0.038527      0.038102
1      1000000.0             10.0      10000.0    0.039967    0.038420    0.038332      0.038906
2      1000000.0             30.0      10000.0    0.039563    0.042514    0.039235      0.040437
3      1000000.0            100.0      10000.0    0.040159    0.040470    0.040262      0.040297
4      1000000.0            300.0      10000.0    0.041472    0.040905    0.041262      0.041213
5      1000000.0           1000.0      10000.0    0.042120    0.042336    0.041970      0.042142
6      1000000.0          10000.0      10000.0    0.042723    0.045244    0.043903      0.043957
7      1000000.0         100000.0      10000.0    0.044796    0.044508    0.044926      0.044743
8      1000000.0        1000000.0      10000.0    0.045805    0.046835    0.048542      0.047060
INFO:root:get_segment_num_from_sba_index: 'forward' strand
INFO:root:
   total_seq_len  num_chromosomes  num_lookups  run_time_0  run_time_1  run_time_2  avg_run_time
0      1000000.0              1.0      10000.0    0.039201    0.039037    0.038942      0.039060
1      1000000.0             10.0      10000.0    0.039330    0.039358    0.039618      0.039435
2      1000000.0             30.0      10000.0    0.040522    0.040198    0.040324      0.040348
3      1000000.0            100.0      10000.0    0.041110    0.041332    0.041207      0.041216
4      1000000.0            300.0      10000.0    0.041789    0.042033    0.042119      0.041981
5      1000000.0           1000.0      10000.0    0.043055    0.043547    0.043210      0.043271
6      1000000.0          10000.0      10000.0    0.043957    0.044299    0.043353      0.043870
7      1000000.0         100000.0      10000.0    0.045026    0.044965    0.045974      0.045322
8      1000000.0        1000000.0      10000.0    0.049351    0.045875    0.045628      0.046951
INFO:root:get_segment_num_from_sba_index: 'both' strand
INFO:root:
   total_seq_len  num_chromosomes  num_lookups  run_time_0  run_time_1  run_time_2  avg_run_time
0      1000000.0              1.0      10000.0    0.037638    0.038248    0.042015      0.039300
1      1000000.0             10.0      10000.0    0.040997    0.041146    0.041057      0.041067
2      1000000.0             30.0      10000.0    0.040527    0.041236    0.050133      0.043965
3      1000000.0            100.0      10000.0    0.040708    0.040615    0.040547      0.040623
4      1000000.0            300.0      10000.0    0.040943    0.041059    0.041096      0.041033
5      1000000.0           1000.0      10000.0    0.041871    0.041370    0.041676      0.041639
6      1000000.0          10000.0      10000.0    0.042773    0.042789    0.043070      0.042877
7      1000000.0         100000.0      10000.0    0.044674    0.044517    0.044587      0.044593
8      1000000.0        1000000.0      10000.0    0.045328    0.045773    0.046894      0.045998
INFO:root:Total init profiling run time: 192.78268957138062
INFO:root:Total get_segment_num_from_sba_index run time: 278.0483183860779
```