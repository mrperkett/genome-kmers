import tempfile
import time
from typing import Union

import numpy as np
import pandas as pd

from genome_kmers.kmers import Kmers
from genome_kmers.sequence_collection import SequenceCollection


def get_random_seq(seq_len: int) -> str:
    """
    Get a random sequence of length seq_len.

    Args:
        seq_len (int): sequence length

    Returns:
        str: sequence
    """
    bases = np.array(["A", "T", "G", "C"], dtype="U1")
    seq = "".join(np.random.choice(bases, seq_len, replace=True))
    return seq


def get_random_seq_list(total_seq_len: int, num_chromosomes: int) -> list[tuple[str, str]]:
    """
    Generate a randomized seq_list that can be used as input for SequenceCollection initialization.

    Args:
        total_seq_len (int): total length of all sequences combined
        num_chromosomes (int): total number of chromosomes into which total_seq_len is divided

    Returns:
        list[tuple[str, str]]: seq_list
    """
    if total_seq_len < num_chromosomes:
        raise ValueError(f"total_seq_len ({total_seq_len}) < num_chromosomes ({num_chromosomes})")
    if num_chromosomes < 1:
        raise ValueError(f"num_chromosomes ({num_chromosomes}) < 1")

    avg_chromosome_length = total_seq_len // num_chromosomes
    seq_list = []
    for chromosome_num in range(num_chromosomes):
        if chromosome_num == num_chromosomes - 1:
            chromosome_length = total_seq_len - (num_chromosomes - 1) * avg_chromosome_length
        else:
            chromosome_length = avg_chromosome_length
        record_name = f"chr{chromosome_num}"
        seq = get_random_seq(chromosome_length)
        seq_list.append((record_name, seq))
    return seq_list


def get_run_time(func):
    """
    Decorator that returns the run time for a single call to func
    """

    def wrapper(*args, **kv):

        start_time = time.time()
        func(*args, **kv)
        return time.time() - start_time

    return wrapper


@get_run_time
def run_seq_list_init(seq_list: list[tuple[str, str]], strand_to_load: str):
    """
    Function to be run when profiling seq_list initialization of a SequenceCollection

    Args:
        seq_list (list[tuple[str, str]]): sequence list input
        strand_to_load (str): strand to load ("forward", "reverse_complement", "both")
    """
    SequenceCollection(sequence_list=seq_list, strands_to_load=strand_to_load)


@get_run_time
def run_get_segment_num_from_sba_index(
    seq_coll: SequenceCollection, sba_strand: str, sba_indices: np.array
):
    """
    Function to be run when profiling get_segment_num_from_sba_index.

    Args:
        seq_coll (SequenceCollection): sequence collection object
        sba_strand (str): strand to use for lookups
        sba_indices (np.array): list of sequence byte array indices for which to do segment_num
            lookups
    """
    for sba_index in sba_indices:
        seq_coll.get_segment_num_from_sba_index(sba_index, sba_strand)


@get_run_time
def run_fasta_init(fasta_file_path, strand):
    SequenceCollection(fasta_file_path=fasta_file_path, strands_to_load=strand)


@get_run_time
def run_kmers_sort(kmers):
    kmers.sort()


def profile_seq_list_init(
    total_seq_len_list: list[Union[int, float]] = [1e4, 1e5, 1e6, 1e7, 1e8],
    num_chromosomes: int = 10,
    num_iterations: int = 3,
    strand_to_load: str = "forward",
    seed: int = 42,
    discard_first_run: bool = True,
) -> pd.DataFrame:
    """
    Profile SequenceCollection initialization using a seq_list input.

    Args:
        total_seq_len_list (list[Union[int, float]], optional): total sequence length. Defaults
            to [1e4, 1e5, 1e6, 1e7, 1e8].
        num_chromosomes (int, optional): number of chromosomes into which sequence is divided.
            Defaults to 10.
        num_iterations (int, optional): number of run_times over which to average. Defaults to 3.
        strand_to_load (str, optional): strand to load. Defaults to "forward".
        seed (int, optional): seed for random number generator. Defaults to 42.
        discard_first_run (bool, optional): whether to disregard the first run when profiling. This
            is useful when numba JIT is used and takes significantly longer the first time a
            function is called and compiled. Defaults to True.

    Returns:
        pd.DataFrame: pandas dataframe with summary stats
    """
    # convert all values to integers if they haven't been already (e.g. if specified as 1e6)
    total_seq_len_list = [int(val) for val in total_seq_len_list]

    # run profiling saving run times
    np.random.seed(seed)
    profiling_data = np.zeros((len(total_seq_len_list), num_iterations + 1), dtype=float)
    for i, total_seq_len in enumerate(total_seq_len_list):
        if discard_first_run:
            # run once without tracking timing since numba compile throws it off
            seq_list = get_random_seq_list(total_seq_len, num_chromosomes)
            run_seq_list_init(seq_list, strand_to_load)

        profiling_data[i, 0] = total_seq_len
        for iter_num in range(num_iterations):
            seq_list = get_random_seq_list(total_seq_len, num_chromosomes)
            run_time = run_seq_list_init(seq_list, strand_to_load)
            profiling_data[i, iter_num + 1] = run_time

    # construct a dataframe from the profiling data
    columns = ["total_seq_len"] + [f"run_time_{i}" for i in range(num_iterations)]
    profiling_df = pd.DataFrame(profiling_data, columns=columns)
    profiling_df["avg_run_time"] = profiling_df.iloc[:, 1:].mean(axis=1)

    return profiling_df


def get_sba_indices_to_test(
    seq_coll: SequenceCollection, num_lookups: int, strand: str, shuffle: bool = True
) -> np.array:
    """
    Get an array with sequence byte array indices to test.

    Args:
        seq_coll (SequenceCollection): SequenceCollection object
        num_lookups (int): length of sequence byte array indices to return
        strand (str): "forward" or "reverse_complement"
        shuffle (bool, optional): whether to shuffle the order of the indices before returning.
            Defaults to True.

    Returns:
        np.array: sba_indices (dtype=np.uint32, length=num_lookups)
    """
    if num_lookups < 1:
        raise ValueError(f"num_lookups ({num_lookups}) < 1")

    # start by adding at least one index from every chromosome
    if strand == "forward":
        sba_seg_starts = seq_coll._forward_sba_seg_starts
        sba_len = len(seq_coll.forward_sba)
    elif strand == "reverse_complement":
        sba_seg_starts = seq_coll._revcomp_sba_seg_starts
        sba_len = len(seq_coll.revcomp_sba)
    else:
        raise ValueError(f"strand ({strand}) not recognized")

    sba_indices = sba_seg_starts.copy()

    # collect additional sba indices to total to num_lookups total
    if len(sba_indices) >= num_lookups:
        # if we have exceeded num_lookups, return the first num_lookups
        sba_indices = sba_indices[:num_lookups]
    else:
        # if we still have additional num_lookups, add them linearly spaced between the min and max
        num_left = num_lookups - len(sba_indices)
        additional_sba_indices = np.linspace(0, sba_len - 1, num_left, dtype=np.uint32)
        sba_indices = np.concatenate((sba_indices, additional_sba_indices))

    # shuffle array order
    if shuffle:
        np.random.shuffle(sba_indices)

    if len(sba_indices) != num_lookups:
        raise AssertionError("Logical error. len(sba_indices) != num_lookups")

    return sba_indices


def profile_get_segment_num_from_sba_index(
    num_chromosomes_list: list[int] = [1, 10, 100, 1000, int(1e4), int(1e5), int(1e6)],
    total_seq_len: int = int(1e6),
    num_iterations: int = 3,
    strand_to_load: str = "forward",
    num_lookups: int = int(1e3),
    seed: int = 42,
    discard_first_run: bool = True,
) -> pd.DataFrame:
    """
    Profile SequenceCollection segment number lookup from a sequence byte array index.

    Args:
        num_chromosomes_list (list[int], optional): list of number of chromosomes into which the
            sequence is divided at each step of the profiling. Defaults to [1, 10, 100, 1000,
            int(1e4), int(1e5), int(1e6)].
        num_iterations (int, optional): total sequence length. Defaults to 3.
        strand_to_load (str, optional): strand to load. Defaults to "forward".
        num_lookups (int, optional): number of times a segment number is looked up from a sequence
            byte array index during each profiling step. Defaults to int(1e3).
        seed (int, optional): seed for random number generator. Defaults to 42.
        discard_first_run (bool, optional): whether to disregard the first run when profiling. This
            is useful when numba JIT is used and takes significantly longer the first time a
            function is called and compiled. Defaults to True.

    Returns:
        pd.DataFrame: pandas dataframe with summary stats
    """
    # set strand_to_test, which is the strand on which the sequence byte array index is defined
    if strand_to_load == "forward" or strand_to_load == "both":
        strand_to_test = "forward"
    else:
        strand_to_test = "reverse_complement"

    # run profiling saving run times
    np.random.seed(seed)
    profiling_data = np.zeros((len(num_chromosomes_list), num_iterations + 3), dtype=float)
    for i, num_chromosomes in enumerate(num_chromosomes_list):
        if discard_first_run:
            # run once without tracking timing since numba compile throws it off
            seq_list = get_random_seq_list(total_seq_len, num_chromosomes)
            seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load=strand_to_load)
            sba_indices = get_sba_indices_to_test(seq_coll, num_lookups, strand_to_test)
            run_get_segment_num_from_sba_index(seq_coll, strand_to_test, sba_indices)

        profiling_data[i, :3] = [total_seq_len, num_chromosomes, num_lookups]
        for iter_num in range(num_iterations):
            seq_list = get_random_seq_list(total_seq_len, num_chromosomes)
            seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load=strand_to_load)
            sba_indices = get_sba_indices_to_test(seq_coll, num_lookups, strand_to_test)
            run_time = run_get_segment_num_from_sba_index(seq_coll, strand_to_test, sba_indices)
            profiling_data[i, iter_num + 3] = run_time

    # construct a dataframe from the profiling data
    columns = ["total_seq_len", "num_chromosomes", "num_lookups"] + [
        f"run_time_{i}" for i in range(num_iterations)
    ]
    profiling_df = pd.DataFrame(profiling_data, columns=columns)
    profiling_df["avg_run_time"] = profiling_df.iloc[:, 3:].mean(axis=1)

    return profiling_df


def write_seq_list_to_file(seq_list: list[tuple[str, str]], out_file, max_line_length: int = 80):
    """
    Write seq_list to out_file (file-like object) with a maximum line length before line break of
    max_line_length.

    Args:
        seq_list (list[tuple[str, str]]): list of sequence ID and sequence string pairs
        out_file (file-like object): output file to which to write
        max_line_length (int, optional): Maximum line length before a new line. Defaults to 80.

    Raises:
        ValueError: raised if any record_id is longer than max_line_length since it cannot be
            wrapped like a sequence could
    """
    for record_id, seq in seq_list:
        if len(record_id) > max_line_length:
            raise ValueError(
                f"record_id ({record_id}) is longer than max_line_length ({max_line_length})"
            )
        out_file.write(f">{record_id}\n")
        for start in range(0, len(seq), max_line_length):
            end = min(start + max_line_length, len(seq))
            out_file.write(seq[start:end])
            out_file.write("\n")


def profile_fasta_init(
    total_seq_lengths: list[int] = [1000, 10000, 100000],
    num_chromosomes: int = 10,
    max_line_length: int = 80,
    strand: str = "forward",
    num_iterations: int = 3,
    seed: int = 42,
    discard_first_run: bool = True,
) -> pd.DataFrame:
    """
    Profile SequenceCollection initialization by fasta file.

    Args:
        total_seq_lengths (list[int], optional): total sequence length for generated fasta file.
            Defaults to [1000, 10000, 100000].
        num_chromosomes (int, optional): number of chromosomes for generated fasta file. Defaults
            to 10.
        max_line_length (int, optional): maximum line length before a new line. Defaults to 80.
        strand (str, optional): strand to load during fasta init. Defaults to "forward".
        num_iterations (int, optional): number of iterations for profiling. Defaults to 3.
        seed (int, optional): random number generator seed. Defaults to 42.
        discard_first_run (bool, optional): whether to disregard the first run when profiling. This
            is useful when numba JIT is used and takes significantly longer the first time a
            function is called and compiled. Defaults to True.

    Returns:
        pd.DataFrame: pandas dataframe with summary stats
    """
    np.random.seed(seed)
    profiling_data = []
    for total_seq_len in total_seq_lengths:

        num_iterations_to_run = num_iterations + 1 if discard_first_run else num_iterations
        row = [total_seq_len, num_chromosomes, max_line_length]
        for iter_num in range(num_iterations_to_run):

            # get a randomly generated sequence list with the desired length and number of
            # chromosomes
            seq_list = get_random_seq_list(total_seq_len, num_chromosomes)

            with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_file:
                # write seq_list to temporary fasta file
                write_seq_list_to_file(seq_list, temp_file, max_line_length=max_line_length)
                temp_file.close()

                # run fasta init
                fasta_file_path = temp_file.name
                run_time = run_fasta_init(fasta_file_path, strand=strand)

            # run once without tracking timing since numba compile throws it off
            if discard_first_run and iter_num == 0:
                continue
            else:
                row.append(run_time)
        profiling_data.append(row)

    # construct a dataframe from the profiling data
    columns = ["total_seq_len", "num_chromosomes", "max_line_length"] + [
        f"run_time_{i}" for i in range(num_iterations)
    ]
    profiling_df = pd.DataFrame(profiling_data, columns=columns)
    profiling_df["avg_run_time"] = profiling_df.iloc[:, 3:].mean(axis=1)

    return profiling_df


def profile_kmers_sort(
    total_seq_lengths: list[int] = [1000, 10000, 100000],
    min_kmer_len: int = 1,
    max_kmer_lens: list[int] = [20, 30, 50, None],
    num_chromosomes: int = 10,
    max_line_length: int = 80,
    strand: str = "forward",
    num_iterations: int = 3,
    seed: int = 42,
    discard_first_run: bool = True,
) -> pd.DataFrame:
    """
    Profile Kmers sort.

    Args:
        total_seq_lengths (list[int], optional): total sequence length for generated fasta file.
            Defaults to [1000, 10000, 100000].
        min_kmer_len (int, optional): shortest allowed kmer length. Defaults to 1.
        max_kmer_lens (list[int], optional): list of max allowed kmer sizes to test. Defaults to
            [20, 30, 50, None].
        num_chromosomes (int, optional): number of chromosomes for generated fasta file. Defaults
            to 10.
        max_line_length (int, optional): maximum line length before a new line. Defaults to 80.
        strand (str, optional): strand to load during fasta init. Defaults to "forward".
        num_iterations (int, optional): number of iterations for profiling. Defaults to 3.
        seed (int, optional): random number generator seed. Defaults to 42.
        discard_first_run (bool, optional): whether to disregard the first run when profiling. This
            is useful when numba JIT is used and takes significantly longer the first time a
            function is called and compiled. Defaults to True.

    Returns:
        pd.DataFrame: pandas dataframe with summary stats
    """
    np.random.seed(seed)
    profiling_data = []
    num_iterations_to_run = num_iterations + 1 if discard_first_run else num_iterations
    for total_seq_len in total_seq_lengths:
        # get a randomly generated sequence list with the desired length and number of
        # chromosomes
        seq_list = get_random_seq_list(total_seq_len, num_chromosomes)
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_file:
            # write seq_list to temporary fasta file
            write_seq_list_to_file(seq_list, temp_file, max_line_length=max_line_length)
            temp_file.close()

            # initialize SequenceCollection and Kmers
            fasta_file_path = temp_file.name
            seq_coll = SequenceCollection(fasta_file_path=fasta_file_path, strands_to_load=strand)

            # run profiling varying max_kmer_len
            for max_kmer_len in max_kmer_lens:
                row = [total_seq_len, num_chromosomes, max_line_length, min_kmer_len, max_kmer_len]
                kmers = Kmers(
                    seq_coll,
                    min_kmer_len=min_kmer_len,
                    max_kmer_len=max_kmer_len,
                    source_strand="forward",
                    track_strands_separately=False,
                )
                for iter_num in range(num_iterations_to_run):

                    run_time = run_kmers_sort(kmers)

                    # run once without tracking timing since numba compile throws it off
                    if discard_first_run and iter_num == 0:
                        continue
                    else:
                        row.append(run_time)
                profiling_data.append(row)

    # construct a dataframe from the profiling data
    columns = [
        "total_seq_len",
        "num_chromosomes",
        "max_line_length",
        "min_kmer_len",
        "max_kmer_len",
    ] + [f"run_time_{i}" for i in range(num_iterations)]
    profiling_df = pd.DataFrame(profiling_data, columns=columns)
    profiling_df["avg_run_time"] = profiling_df.iloc[:, 5:].mean(axis=1)

    return profiling_df
