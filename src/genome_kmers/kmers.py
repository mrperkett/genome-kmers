import shelve
from pathlib import Path
from typing import Callable, Generator, Union

import h5py
import numba as nb
import numpy as np
from numba import jit
from numba.misc import quicksort

from genome_kmers.sequence_collection import SequenceCollection


@jit
def kmer_filter_keep_all(sba: np.array, sba_strand: str, kmer_sba_start_idx: int):
    return True


def gen_kmer_length_filter_func(min_kmer_len: int) -> Callable:
    """
    Generates a filter that passes if kmer is of min_kmer_len, but otherwise fails.

    Args:
        min_kmer_len (int): minimum required kmer length

    Returns:
        Callable: filter
    """

    @jit
    def filter(sba: np.array, sba_strand: str, kmer_sba_start_idx: int) -> bool:
        return kmer_has_required_len(sba, kmer_sba_start_idx, min_kmer_len)

    return filter


def gen_kmer_homopolymer_filter_func(max_homopolymer_size: int, kmer_len: int) -> Callable:
    """
    Generate a filter function that passes if there is no homopolym of length greater than
    max_homopolymer_size.

    NOTE: this function does not do extra checks to ensure that the kmer itself is valid (i.e. it
    does overflow over a boundary or beyond the sequence byte array, is made of valid bases, etc).

    Args:
        max_homopolymer_size (int): largest allowed homopolymer.  Must be >= 1
        kmer_len (int): length of kmer

    Raises:
        ValueError: max_homopolymer_size must be >= 2

    Returns:
        Callable: filter
    """
    # verify that max_homopolymer is valid
    if max_homopolymer_size < 1:
        raise ValueError(f"max_homopolymer_size ({max_homopolymer_size}) must be >= 1")

    # verify that kmer_len is valid
    if kmer_len < 1:
        raise ValueError(f"kmer_len ({kmer_len}) must be >= 1")

    @jit
    def filter(sba: np.array, sba_strand: str, kmer_sba_start_idx: int):
        # verify that a kmer starting at kmer_sba_start_idx and of length kmer_len does not overflow
        if kmer_sba_start_idx + kmer_len - 1 >= len(sba):
            raise ValueError(
                f"The kmer_len ({kmer_len}) requested is too large for kmer_sba_start_idx ({kmer_sba_start_idx})"
            )

        # if the length of the kmer is less than the max allowed homopolymer, we don't need to do
        # any checks
        if kmer_len < max_homopolymer_size:
            return True

        # iterate base by base counting up the homopolymer size
        homopolymer_size = 1
        for kmer_idx in range(1, kmer_len):
            sba_idx = kmer_sba_start_idx + kmer_idx
            base = sba[sba_idx]
            prev_base = sba[sba_idx - 1]

            if base == ord("$"):
                raise ValueError(
                    f"The kmer_len ({kmer_len}) requested is too large for kmer_sba_start_idx ({kmer_sba_start_idx})"
                )

            # if part of a homopolymer, increment homopolymer_size
            if base == prev_base:
                homopolymer_size += 1
                # if homopolymer size has exceeded the max, return False
                if homopolymer_size > max_homopolymer_size:
                    return False
            # otherwise, reset
            else:
                homopolymer_size = 1

        return True

    return filter


def gen_kmer_gc_content_filter_func(
    min_allowed_gc_frac: float,
    max_allowed_gc_frac: float,
    kmer_len: int,
) -> Callable:
    """
    Generate a filter function that returns True if fraction GC content is between
    min_allowed_gc_frac and max_allowed_gc_frac.

    NOTE: this function does not do extra checks to ensure that the kmer itself is valid (i.e. it
    does overflow over a boundary or beyond the sequence byte array, is made of valid bases, etc).

    Args:
        min_allowed_gc_frac (float): minimum allowed fraction GC content (inclusive). Must be in
            the range [0.0, 1.0] and be <= max_allowed_gc_frac.
        max_allowed_gc_frac (float): maximum allowed fraction GC content (inclusive). Must be in
            the range [0.0, 1.0] and be >= min_allowed_gc_frac.
        kmer_len (int): length of kmer

    Raises:
        ValueError: min_allowed_gc_frac or max_allowed_gc_frac is invalid

    Returns:
        Callable: filter
    """
    # check that min and max_allowed_gc_frac are valid
    if min_allowed_gc_frac > max_allowed_gc_frac:
        raise ValueError(
            f"min_allowed_gc_frac ({min_allowed_gc_frac}) must be <= max_allowed_gc_frac ({max_allowed_gc_frac})"
        )
    if min_allowed_gc_frac < 0.0 or min_allowed_gc_frac > 1.0:
        raise ValueError(
            f"min_allowed_gc_frac ({min_allowed_gc_frac}) must be in the range [0.0, 1.0]"
        )
    if max_allowed_gc_frac < 0.0 or max_allowed_gc_frac > 1.0:
        raise ValueError(
            f"max_allowed_gc_frac ({max_allowed_gc_frac}) must be in the range [0.0, 1.0]"
        )

    # calculate the min and max GC counts allowed
    min_allowed_gc_count = int(np.ceil(kmer_len * min_allowed_gc_frac))
    max_allowed_gc_count = int(np.floor(kmer_len * max_allowed_gc_frac))

    # store values for "G" and "C"
    g_val = ord("G")
    c_val = ord("C")

    @jit
    def filter(sba: np.array, sba_strand: str, kmer_sba_start_idx: int):
        # if it's not possible to meet the fraction requirements, return False
        #
        # Example of the edge case where two fractions are so close that it's not possible to meet
        # the requirements
        #   kmer_len = 16
        #   min_allowed_gc_count = 0.13
        #   max_allowed_gc_count= 0.18
        #
        #   0.18 * 16 = 2.88
        #   min_allowed_gc_count = ceil(0.13 * 16)
        #                        = ceil(2.08)
        #                        = 3
        #   max_allowed_gc_count = floor(0.18 * 16)
        #                        = floor(2.88)
        #                        = 2
        if max_allowed_gc_count < min_allowed_gc_count:
            return False

        # loop through kmer counting the GC bases
        gc_count = 0
        for kmer_idx in range(kmer_len):
            sba_idx = kmer_sba_start_idx + kmer_idx
            base = sba[sba_idx]

            if base == ord("$"):
                raise ValueError(
                    f"The kmer_len ({kmer_len}) requested is too larger for kmer_sba_start_idx ({kmer_sba_start_idx})"
                )

            if base == g_val or base == c_val:
                gc_count += 1
                # if we have already exceeded the max allowed GC counts, return False
                if gc_count > max_allowed_gc_count:
                    return False

        # if the GC count is within the required range, return True
        if min_allowed_gc_count <= gc_count and gc_count <= max_allowed_gc_count:
            return True
        return False

    return filter


def gen_no_ambiguous_bases_filter(kmer_len: int) -> Callable:
    """
    Generate a filter that checks that only "A", "T", "G", or "C" are present in the kmer.

    Args:
        kmer_len (int): kmer length

    Raises:
        ValueError: end of segment or sequence byte array is reached before end of kmer

    Returns:
        Callable: no_ambiguous_bases_filter
    """

    @jit
    def no_ambiguous_bases_filter(sba: np.array, sba_strand: str, kmer_sba_start_idx: int):
        # check that kmer_len is valid given kmer_sba_start_idx
        if kmer_sba_start_idx + kmer_len > len(sba):
            raise ValueError(f"kmer_len ({kmer_len}) is invalid. It extends beyond len(sba)")

        passes = True
        for i in range(kmer_len):
            base = sba[kmer_sba_start_idx + i]
            # check whether the end of the segment has been reached
            # ord("$") == 36
            if base == 36:
                raise ValueError(f"end of segment was reached. kmer_len ({kmer_len}) invalid.")
            # check if it's not any of A, T, G, C
            # ord("A") == 65, ord("T") == 84, ord("G") == 71, ord("C") == 67
            if base != 65 and base != 84 and base != 71 and base != 67:
                passes = False
                break
        return passes

    return no_ambiguous_bases_filter


@jit
def crispr_ngg_pam_filter(sba: np.array, sba_strand: str, kmer_sba_start_idx: int) -> bool:
    """
    Generate a filter that passes for all 23-mers ending in GG.

    NOTE: no other checks on kmer validity are carried out.

    Raises:
        ValueError: kmer extend beyond the size of the sequence byte array

    Returns:
        bool: whether kmer passes filter or not
    """
    # diagram of a SpyCas9 CRISPR guide
    # 0                  19
    # |                  |
    # --------------------NGG
    # [      target      ]PAM

    # check that arguments are valid
    if kmer_sba_start_idx + 23 > len(sba):
        raise ValueError("The guide defined at this start index extends beyond the sba")

    # check whether there is an NGG PAM
    # ord("G") == 71
    if sba[kmer_sba_start_idx + 21] == 71 and sba[kmer_sba_start_idx + 22] == 71:
        return True
    return False


@jit
def kmer_has_required_len(sba: np.array, sba_start_idx: int, min_kmer_len: int) -> bool:
    """
    Checks whether the kmer is of at least min_kmer_len before reaching the end of the segment.

    Args:
        sba (np.array): sequence byte array
        sba_start_idx (int): sequence byte array start index for the kmer
        min_kmer_len (int): minimum kmer length

    Returns:
        bool:
    """
    for idx in range(sba_start_idx, sba_start_idx + min_kmer_len):
        # determine whether the kmer has reached the end of a segment
        idx_is_at_end_of_segment = True if idx >= len(sba) or sba[idx] == ord("$") else False

        # if end of segment has been reached, the kmer is invalid
        if idx_is_at_end_of_segment:
            return False
    return True


def get_compare_sba_kmers_func(kmer_len):
    @jit
    def compare_sba_kmers_func(sba_a, sba_b, kmer_sba_start_idx_a, kmer_sba_start_idx_b):
        return compare_sba_kmers_lexicographically(
            sba_a, sba_b, kmer_sba_start_idx_a, kmer_sba_start_idx_b, max_kmer_len=kmer_len
        )

    return compare_sba_kmers_func


@jit
def compare_sba_kmers_always_less_than(
    sba_a: np.array,
    sba_b: np.array,
    kmer_sba_start_idx_a: int,
    kmer_sba_start_idx_b: int,
    max_kmer_len: Union[int, None] = None,
) -> tuple[int, int]:
    return -1, 0


@jit
def compare_sba_kmers_lexicographically(
    sba_a: np.array,
    sba_b: np.array,
    kmer_sba_start_idx_a: int,
    kmer_sba_start_idx_b: int,
    max_kmer_len: Union[int, None] = None,
) -> tuple[int, int]:
    """
    Lexicographically compare two kmers of length kmer_len.  If kmer_len is None, the end of the
    segment defines the longest kmer.

    NOTE: This function does no validation for kmer_len. It will compare up to max_kmer_len bases
    if required, but it will return as soon as the comparison result is known.

    Args:
        sba_a (np.array): sequence byte array a
        sba_b (np.array): sequence byte array b
        kmer_sba_start_idx_a (int): index in sba that defines the start of kmer a
        kmer_sba_start_idx_b (int): index in sba that defines the start of kmer b
        kmer_len (Union[int, None], optional): Length of the kmers to compare.  If None, the
            end of the segment defines the longest kmers to compare.. Defaults to None.

    Raises:
        AssertionError: there were no valid bases to compare

    Returns:
        tuple[int, int]: comparison, last_kmer_index_compared
            comparison:
                +1 = kmer_a > kmer_b
                0 = kmer_a == kmer_b
                -1 = kmer_a < kmer_b
            last_kmer_index_compared: the kmer index of the last valid comparison done between two
                bases.  If a single base was compare, then this value will be 0.
    """
    # Example
    #
    # str(sba):    ATGGGCTGCAAGCTCGA$AATTTAGCGGCCTAGGCTTA
    # kmer_a:             [--------]
    # kmer_b:                 [----]
    #
    # max_kmer_len  |   comparison
    # 1             |   0
    # 2             |   0
    # 3             |   -1
    # None          |   -1
    kmer_idx = 0
    comparison = 0
    last_kmer_index_compared = None
    while True:
        # sba indices to compare
        idx_a = kmer_sba_start_idx_a + kmer_idx
        idx_b = kmer_sba_start_idx_b + kmer_idx

        # is idx_a or idx_b out of bounds? (i.e. equal to "$" or overflowed)
        idx_a_out_of_bounds = True if idx_a >= len(sba_a) or sba_a[idx_a] == ord("$") else False
        idx_b_out_of_bounds = True if idx_b >= len(sba_b) or sba_b[idx_b] == ord("$") else False

        # break if idx_a or idx_b is out of bounds
        if idx_a_out_of_bounds or idx_b_out_of_bounds:
            # set last_kmer_index_compared
            last_kmer_index_compared = kmer_idx - 1
            if last_kmer_index_compared < 0:
                raise AssertionError(f"There were no valid kmer bases to compare")

            # get comparison
            if idx_a_out_of_bounds and not idx_b_out_of_bounds:
                comparison = -1  # kmer_a < kmer_b
            elif idx_b_out_of_bounds and not idx_a_out_of_bounds:
                comparison = 1  # kmer_a > kmer_b
            else:
                comparison = 0  # kmer_a == kmer_b
            break

        # check whether kmer_a < kmer_b (and vice versa)
        if sba_a[idx_a] < sba_b[idx_b]:
            comparison = -1  # kmer_a < kmer_b
            last_kmer_index_compared = kmer_idx
            break
        if sba_a[idx_a] > sba_b[idx_b]:
            comparison = 1  # kmer_a > kmer_b
            last_kmer_index_compared = kmer_idx
            break

        # break if kmer_len has been reached
        if max_kmer_len is not None and kmer_idx == max_kmer_len - 1:
            last_kmer_index_compared = kmer_idx
            break

        kmer_idx += 1

    return comparison, last_kmer_index_compared


@jit
def get_kmer_info_minimal(
    kmer_num: int,
    kmer_sba_start_indices: np.array,
    sba: np.array,
    kmer_len: Union[int, None],
    group_size_yielded: int,
    group_size_total: int,
) -> tuple[int, int, int]:
    """
    Return only basic kmer information without any processing. Used as an input to
    kmer_info_by_group_generator when only basic information is required.

    Args:
        kmer_num (int): kmer number
        kmer_start_indices (np.array): kmer sequence byte array start indices
        sba (np.array): sequence byte array
        kmer_len (Union[int, None]): length of kmer.  If None, take the longest possible.
        group_size_yielded (int): number of kmers in the group that will be yielded
        group_size_total (int): number of kmers in the group in total

    Returns:
        tuple[int, int, int]: kmer_num, group_size_yielded, group_size_total
    """

    return kmer_num, group_size_yielded, group_size_total


@jit
def get_kmer_info_group_size_only(
    kmer_num: int,
    kmer_sba_start_indices: np.array,
    sba: np.array,
    kmer_len: Union[int, None],
    group_size_yielded: int,
    group_size_total: int,
) -> tuple[int, int, int]:
    """
    Return only group_size_total without any processing.

    Args:
        kmer_num (int): kmer number
        kmer_start_indices (np.array): kmer sequence byte array start indices
        sba (np.array): sequence byte array
        kmer_len (Union[int, None]): length of kmer.  If None, take the longest possible.
        group_size_yielded (int): number of kmers in the group that will be yielded
        group_size_total (int): number of kmers in the group in total

    Returns:
        int: group_size_total
    """
    return group_size_total


@jit
def get_kmer_group_size_hist(
    sba: np.array,
    sba_strand: str,
    kmer_len: Union[int, None],
    kmer_start_indices: np.array,
    kmer_comparison_func: Callable,
    kmer_filter_func: Callable,
    min_group_size: int = 1,
    max_group_size: Union[int, None] = None,
    max_counts_bin: int = 1000000,
) -> tuple[np.array, int]:
    """
    Build a histogram of group sizes.  counts_by_group_size[i] gives the number of groups of size
    i.  Any group sizes > max_counts_bin will be placed in max_counts_bin.  The total number of
    kmers is also calculated.

    Args:
        sba (np.array): sequence byte array
        sba_strand (str): "forward" or "reverse_complement"
        kmer_len (Union[int, None]): length of kmer.  If None, take the longest possible.
        kmer_start_indices (np.array): kmer sequence byte array start indices
        kmer_comparison_func (Callable): function that returns the result of a two kmer comparison
        kmer_filter_func (Callable): function that returns true if a kmer passes all filters
        min_group_size (int, optional): Smallest allowed group size. Defaults to 1.
        max_group_size (Union[int, None], optional): Largest allowed group size.  If None, then
            there is no maximum group size. Defaults to None.
        max_counts_bin (int, optional): largest group_size bin in the return counts_by_group_size
            array. Group sizes that exceed this value will be placed in this bin. Defaults to
            1000000.

    Raises:
        ValueError: max_counts_bin is invalid

    Returns:
        tuple[np.array, int]: counts_by_group_size, total_kmer_count
    """
    if max_counts_bin <= 0:
        raise ValueError(f"max_counts_bin ({max_counts_bin}) must be >= 1")

    # set kmer_info_func and yield_first_n so that the group size is yielded once per group
    kmer_info_func = get_kmer_info_group_size_only
    yield_first_n = 1

    # initialize the kmer info generator
    kmer_generator = kmer_info_by_group_generator(
        sba,
        sba_strand,
        kmer_len,
        kmer_start_indices,
        kmer_comparison_func,
        kmer_filter_func,
        kmer_info_func,
        min_group_size,
        max_group_size,
        yield_first_n,
    )

    # populate the array to hold the number of group counts for each group size.  Also count the
    # total number of kmers
    counts_by_group_size = np.zeros((max_counts_bin + 1,), dtype=np.int64)
    total_kmer_count = 0
    for group_size_total in kmer_generator:
        total_kmer_count += group_size_total
        counts_by_group_size[min(group_size_total, max_counts_bin)] += 1

    return counts_by_group_size, total_kmer_count


@jit(cache=False)
def kmer_info_by_group_generator(
    sba: np.array,
    sba_strand: str,
    kmer_len: Union[int, None],
    kmer_start_indices: np.array,
    kmer_comparison_func: Callable,
    kmer_filter_func: Callable,
    kmer_info_func: Callable,
    min_group_size: int = 1,
    max_group_size: Union[int, None] = None,
    yield_first_n: Union[int, None] = None,
) -> Generator[tuple, None, None]:
    """
    Generator that yields the valid kmer information and total group size for all groups meeting
    requirements.  A valid kmer is one that passes the provided kmer_filter_func.  A group is
    defined as the set of identical kmers as defined by the kmer_comparison_func. The first
    "yield_first_n" kmers will be yielded if the group meets all provided requirements.  It must
    have a total group size between min_group_size and max_group_size (inclusive). The kmer
    information that is yielded is customizable and defined by kmer_info_func.

    Args:
        sba (np.array): sequence byte array
        sba_strand (str): "forward" or "reverse_complement"
        kmer_len (Union[int, None]): length of kmer.  If None, take the longest possible.
        kmer_start_indices (np.array): kmer sequence byte array start indices
        kmer_comparison_func (Callable): function that returns the result of a two kmer comparison
        kmer_filter_func (Callable): function that returns true if a kmer passes all filters
        kmer_info_func (Callable): function that returns a tuple with all the kmer information to
            yielded.
        min_group_size (int, optional): Smallest allowed group size. Defaults to 1.
        max_group_size (Union[int, None], optional): Largest allowed group size.  If None, then
            there is no maximum group size. Defaults to None.
        yield_first_n (Union[int, None], optional): yield up to this many kmer_nums. Defaults to
            None.

    Raises:
        ValueError: invalide min_group_size, max_group_size, or yield_first_n

    Yields:
        Generator[tuple[list[int], int], None, None]: valid_kmer_nums_in_group, group_size
    """
    # check arguments
    if min_group_size < 1:
        raise ValueError(f"min_group_size ({min_group_size}) must be >= 1")
    if max_group_size is not None and max_group_size < min_group_size:
        raise ValueError(
            f"if max_group_size ({max_group_size}) is specified, it must be >= min_group_size ({min_group_size})"
        )
    if yield_first_n is not None and yield_first_n < 1:
        raise ValueError(f"if yield_first_n ({yield_first_n}) is specified, it must be > 0")

    # iterate through all kmers storing kmers that pass all filters and yielding results when a new
    # group is reached
    # NOTE: the empty list is initialized like it is below so that numba can infer its type
    # https://numba.readthedocs.io/en/stable/user/troubleshoot.html#my-code-has-an-untyped-list-problem
    valid_kmer_nums_in_group = [i for i in range(0)]
    group_size = 0
    prev_valid_kmer_sba_start_idx = None
    for kmer_num in range(0, len(kmer_start_indices)):

        # skip the kmer if it does not pass all filters
        kmer_sba_start_idx = kmer_start_indices[kmer_num]
        passes_all_filters = kmer_filter_func(sba, sba_strand, kmer_sba_start_idx)
        if not passes_all_filters:
            continue

        # if this is the first valid kmer, set prev_valid_kmer_sba_start_idx and treat as if it is
        # in the same group
        if prev_valid_kmer_sba_start_idx is None:
            prev_valid_kmer_sba_start_idx = kmer_sba_start_idx
            in_same_group = True
        # otherwise, check whether we are in the same group by comparing to the last valid kmer
        else:
            comparison, last_kmer_index_compared = kmer_comparison_func(
                sba, sba, prev_valid_kmer_sba_start_idx, kmer_sba_start_idx
            )
            in_same_group = True if comparison == 0 else False
            prev_valid_kmer_sba_start_idx = kmer_sba_start_idx

        # if we are in a the same group, increment group size and add to valid_kmer_nums_in_group
        if in_same_group:
            group_size += 1
            if yield_first_n is None or len(valid_kmer_nums_in_group) < yield_first_n:
                valid_kmer_nums_in_group.append(kmer_num)

        # otherwise, we are in a new group - yield info and start a new group
        else:
            # yield the completed group if it meets requirements
            meets_min_group_size = group_size >= min_group_size
            meets_max_group_size = max_group_size is None or group_size <= max_group_size
            if meets_min_group_size and meets_max_group_size:
                # yield kmer_info for each valid kmer_num
                group_size_yielded = len(valid_kmer_nums_in_group)
                for kmer_num_in_group in valid_kmer_nums_in_group:
                    yield kmer_info_func(
                        kmer_num_in_group,
                        kmer_start_indices,
                        sba,
                        kmer_len,
                        group_size_yielded,
                        group_size,
                    )

            # reset tracking for the new group
            group_size = 1
            valid_kmer_nums_in_group = [kmer_num]

    # there is likely one remaining group to yield (unless there were no valid groups)
    # yield the completed group if it meets requirements
    meets_min_group_size = group_size >= min_group_size
    meets_max_group_size = max_group_size is None or group_size <= max_group_size
    if meets_min_group_size and meets_max_group_size:
        # yield kmer_info for each valid kmer_num
        group_size_yielded = len(valid_kmer_nums_in_group)
        for kmer_num_in_group in valid_kmer_nums_in_group:
            yield kmer_info_func(
                kmer_num_in_group,
                kmer_start_indices,
                sba,
                kmer_len,
                group_size_yielded,
                group_size,
            )

    return


class Kmers:
    """
    Defines memory-efficient kmers calculations on a genome.
    """

    def __init__(
        self,
        seq_coll: Union[SequenceCollection, None] = None,
        min_kmer_len: int = 1,
        max_kmer_len: Union[int, None] = None,
        source_strand: str = "forward",
        track_strands_separately: bool = False,
        method: str = "single_pass",
    ) -> None:
        """
        Initialize Kmers

        Args:
            seq_coll (SequenceCollection): the sequence collection on which kmers are
                defined
            min_kmer_len (int, optional): kmers below this size are not considered.
                Defaults to 1.
            max_kmer_len (tuple[int, None], optional): kmers above this size are not considered.
                Defaults to None.
            source_strand (str, optional): strand(s) on which kmers are defined ("forward",
                "reverse_complement", "both"). Defaults to "forward".
            track_strands_separately (bool, optional): if source_strand is set to "both", this
                specifies whether kmers should be tracked separately by strand (which is
                required for certain kmer filters)
            method (str, optional): which method to use for initialization.  "single_pass" runs
                faster, but temporarily uses more RAM (up to 2x).  "double_pass" runs slower (~2x)
                but uses less memory (as much as half).  Defaults to "single_pass".

        Raises:
            ValueError: invalid arguments
            NotImplementedError: yet to be implemented functionality
        """
        # check whether arguments required functionality that has not yet been implemented
        if track_strands_separately:
            raise NotImplementedError(
                f"This function has not been implemented for track_strands_separately = '{track_strands_separately}'"
            )
        if source_strand != "forward":
            raise NotImplementedError(
                f"This function has not been implemented for source_strand = '{source_strand}'"
            )

        # verify that arguments are valid
        if source_strand not in ("forward", "reverse_complement", "both"):
            raise ValueError(f"source_strand ({source_strand}) not recognized")
        if source_strand != "both" and track_strands_separately:
            raise ValueError(
                f"track_strands_separately can only be true if source_strand is 'both', but it is '{source_strand}'"
            )
        if min_kmer_len < 1:
            raise ValueError(f"min_kmer_len ({min_kmer_len}) must be greater than zero")
        if max_kmer_len is not None:
            if max_kmer_len < 1:
                raise ValueError(f"max_kmer_len ({max_kmer_len}) must be greater than zero")
            if min_kmer_len is not None and max_kmer_len < min_kmer_len:
                raise ValueError(
                    f"max_kmer_len ({max_kmer_len}) is less than min_kmer_len ({min_kmer_len})"
                )

        # set member variables based on arguments
        self.min_kmer_len = min_kmer_len
        self.max_kmer_len = max_kmer_len
        self.kmer_source_strand = source_strand
        self.track_strands_separately = track_strands_separately

        self._is_initialized = False
        self._is_set = False
        self._is_sorted = False
        self.kmer_sba_start_indices = None

        # if seq_coll is not provided, return without any further processing
        if seq_coll is None:
            return

        # count number of records and sequence lengths in sequence_collection
        seq_lengths = []
        min_seq_len = None
        num_records = 0
        record_generator = seq_coll.iter_records()
        for _, sba_seg_start_idx, sba_seg_end_idx in record_generator:
            seq_length = sba_seg_end_idx - sba_seg_start_idx + 1
            seq_lengths.append(seq_length)
            if min_seq_len is None or seq_length < min_seq_len:
                min_seq_len = seq_length
            num_records += 1

        # verify that arguments are valid given the sequence_collection
        if num_records == 0:
            raise ValueError(f"sequence_collection is empty")
        if min_kmer_len is not None and min_kmer_len > min_seq_len:
            raise ValueError(
                f"min_kmer_len ({min_kmer_len}) must be <= the shortest sequence length ({min_seq_len})"
            )
        if seq_coll.strands_loaded() != source_strand:
            # for now, require that source_strand matches what is in the SequenceCollection for ease
            # of implementation
            raise ValueError(
                f"source_strand ({source_strand}) does not match sequence_collection loaded strand ({seq_coll.strands_loaded()})"
            )

        # set seq_coll and initialize
        self.seq_coll = seq_coll
        self._initialize(method=method)

        return

    def _initialize(self, kmer_filters=[], method: str = "single_pass"):
        """
        Initialize Kmers instance and populate internal kmer_sba_start_indices array.

        Args:
            kmer_filters (list, optional): _description_. Defaults to [].
            method (str, optional): which method to use for initialization.  "single_pass" runs
                faster, but temporarily uses more RAM (up to 2x).  "double_pass" runs slower (~2x)
                but uses less memory (as much as half).  Defaults to "single_pass".

        Raises:
            ValueError: method not recognized
        """
        if kmer_filters != []:
            raise NotImplementedError("kmer_filters have not been implemented")

        if method == "double_pass":
            # TODO: "double_pass" implementation counts the number of kmers first, initializes an
            # array of the correct size, and then populates it on-the-fly. Requires less memory
            raise NotImplementedError(f"method '{method}' has not been implemented")
        elif method == "single_pass":
            self._initialize_single_pass(kmer_filters=kmer_filters)
        else:
            raise ValueError(f"method '{method}' not recognized")

        self._is_initialized = True

    def _initialize_single_pass(self, kmer_filters=[]):
        """
        Initialize Kmers in a single pass. This loads all unfiltered kmers into memory, filters
        them in place, and then creates a new array of length num_filtered_kmers. This runs
        faster than a "double pass" initialization, but temporarily uses more memory (up to 2x).

        Args:
            kmer_filters (list, optional): _description_. Defaults to [].

        Raises:
            AssertionError: logic check
        """

        if kmer_filters != []:
            raise NotImplementedError("kmer_filters have not been implemented")

        num_kmers = self._get_unfiltered_kmer_count()
        if num_kmers > 2**32 - 1:
            msg = "the size of the required kmers array exceeds the limit set by a uint32"
            raise NotImplementedError(msg)

        # initialize array large enough to hold all unfiltered kmers
        self.kmer_sba_start_indices = np.zeros(num_kmers, dtype=np.uint32)

        # iterate over records initializing kmer start indices
        record_generator = self.seq_coll.iter_records()
        last_filled_index = -1
        for _, sba_seg_start_idx, sba_seg_end_idx in record_generator:
            num_kmers_in_record = (sba_seg_end_idx - sba_seg_start_idx + 1) - self.min_kmer_len + 1

            kmer_start = last_filled_index + 1
            kmer_end = last_filled_index + 1 + num_kmers_in_record
            sba_start = sba_seg_start_idx
            sba_end = sba_seg_end_idx + 1 - self.min_kmer_len + 1
            self.kmer_sba_start_indices[kmer_start:kmer_end] = np.arange(
                sba_start, sba_end, dtype=np.uint32
            )
            last_filled_index += num_kmers_in_record

        if last_filled_index != num_kmers - 1:
            raise AssertionError(
                f"logic error: last_filled_index ({last_filled_index}) != num_kmers - 1 ({num_kmers - 1})"
            )

        # TODO: next step is to filter kmers in place

        return

    def _get_unfiltered_kmer_count(self) -> int:
        """
        Calculate the number of unfiltered kmers and the total length of the kmer array for the
        loaded sequence_collection.

        Raises:
            ValueError: empty sequence_collection

        Returns:
            int: num_kmers
        """
        # TODO: when SequenceCollection has a method to get num_records, remove the counter below
        num_kmers = 0
        num_records = 0
        record_generator = self.seq_coll.iter_records()
        for _, sba_seg_start_idx, sba_seg_end_idx in record_generator:
            num_kmers_in_record = (sba_seg_end_idx - sba_seg_start_idx + 1) - self.min_kmer_len + 1
            num_kmers += num_kmers_in_record
            num_records += 1

        # TODO: when SequenceCollection has a method to get num_records, move this to top of func
        if num_records == 0:
            raise ValueError(f"SequenceCollection does not have any records")

        return num_kmers

    def __len__(self):
        return len(self.kmer_sba_start_indices)

    def __getitem__(self):
        pass

    def get_kmers(
        self,
        kmer_len: Union[int, None],
        one_based_seq_index: bool = False,
        kmer_filter_func: Callable = kmer_filter_keep_all,
        kmer_info_to_yield: str = "minimum",
        min_group_size: int = 1,
        max_group_size: Union[int, None] = None,
        yield_first_n: Union[int, None] = None,
    ) -> Generator[tuple, None, None]:
        """
        A customizable generator yielding tuples with all kmer information.

        Examples:

        Yield all kmers
        kmers.get_kmers(yield_first_n=None)

        Yield only the first occurrence of a kmer
        kmers.get_kmers(use yield_first_n=1)

        Yield up to the first 10 occurrences of a kmer
        kmers.get_kmers(use yield_first_n=10)

        Yield all kmers that occur exactly once
        kmers.get_kmers(max_group_size=1)

        Yield all kmers that are repeated at least 5 times and no more than 10 times
        kmers.get_kmers(min_group_size=5, max_group_size=10)

        NOTE: group yielding is not supported if kmers are unsorted. The user cannot provide
        min_group_size, max_group_size, or yield_first_n in this situation.

        Args:
            kmer_len (Union[int, None]): length of kmer.  If None, take the longest possible.
            one_based_seq_index (bool, optional): whether yielded sequence indices should be
                1-based. Defaults to False.
            kmer_filter_func (Callable, optional): function that returns true if a kmer passes all
                filters. Defaults to kmer_filter_keep_all.
            kmer_info_to_yield (str): "minimum" or "full". Defaults to "minimum"
            min_group_size (int, optional): Smallest allowed group size. Defaults to 1.
            max_group_size (Union[int, None], optional): Largest allowed group size.  If None, then
                there is no maximum group size. Defaults to None.
            yield_first_n (Union[int, None], optional): yield up to this many kmer_nums. Defaults to
                None.
        Raises:
            NotImplementedError: if kmer_source_strand and seq_coll.strands() loaded are not both
                "forward"
            ValueError: kmer_len is invalid
            ValueError: one or more group params are invalid (min_group_size, max_group_size,
                yield_first_n)
            ValueError: kmer_comparison_func is provided when kmers have not been sorted

        Yields:
            Generator[tuple, None, None]: output depends on get_kmer_info_func
        """

        condition1 = self.kmer_source_strand != "forward"
        condition2 = self.seq_coll.strands_loaded() != "forward"
        if condition1 or condition2:
            raise NotImplementedError(
                f"both kmer_source_strand ({self.kmer_source_strand}) and "
                "sequence_collection.strands_loaded() must be 'forward'"
            )

        # verify that kmer_len is valid
        if kmer_len is not None and kmer_len < 1:
            raise ValueError(f"kmer_len ({kmer_len}) must be > 0")

        # verify that if kmers has not been sorted, that group arguments have not been provided
        if not self._is_sorted:
            if min_group_size != 1:
                msg = "Returning group parameters is not supported when kmers has not been"
                msg += f" sorted. min_group_size ({min_group_size}) cannot be specified. Did you"
                msg += " mean to run sort() before getting kmers?"
                raise ValueError(msg)
            if max_group_size is not None:
                msg = "Returning group parameters is not supported when kmers has not been"
                msg += f" sorted. max_group_size ({max_group_size}) cannot be specified. Did you"
                msg += " mean to run sort() before getting kmers?"
                raise ValueError(msg)
            if yield_first_n is not None:
                msg = "Returning group parameters is not supported when kmers has not been"
                msg += f" sorted. yield_first_n ({yield_first_n}) cannot be specified. Did you"
                msg += " mean to run sort() before getting kmers?"
                raise ValueError(msg)

        # set kmer_comparison_func
        if self._is_sorted:
            kmer_comparison_func = get_compare_sba_kmers_func(kmer_len)
        else:
            kmer_comparison_func = compare_sba_kmers_always_less_than

        # set get_kmer_info_func
        if kmer_info_to_yield == "minimum":
            get_kmer_info_func = get_kmer_info_minimal
        elif kmer_info_to_yield == "full":
            get_kmer_info_func = self.generate_get_kmer_info_func(one_based_seq_index)
        else:
            raise ValueError(f"kmer_info_to_yield ({kmer_info_to_yield}) not recognized")

        # collect values to pass to kmer_info_generator
        sba = self.seq_coll.forward_sba
        sba_strand = self.seq_coll.strands_loaded()
        kmer_start_indices = self.kmer_sba_start_indices

        # initialize the kmer info generator
        kmer_generator = kmer_info_by_group_generator(
            sba,
            sba_strand,
            kmer_len,
            kmer_start_indices,
            kmer_comparison_func,
            kmer_filter_func,
            get_kmer_info_func,
            min_group_size,
            max_group_size,
            yield_first_n,
        )

        for kmer_info in kmer_generator:
            yield kmer_info

        return

    def get_kmer_count(
        self,
        kmer_len: Union[int, None],
        kmer_filter_func: Callable = kmer_filter_keep_all,
        min_group_size: int = 1,
        max_group_size: Union[int, None] = None,
    ) -> int:
        """
        A customizable function to count the total number of kmers passing filters.

        Examples:

        Count kmers that occur exactly once
        kmers.get_kmer_count(max_group_size=1)

        Count kmers that are repeated at least 5 times and no more than 10 times
        kmers.get_kmer_count(min_group_size=5, max_group_size=10)

        Count 50-mers
        filter = gen_kmer_length_filter_func(min_kmer_len=50)
        kmers.get_kmer_count(kmer_filter_func=filter)

        Args:
            kmer_len (Union[int, None]): length of kmer.  If None, take the longest possible.
            kmer_filter_func (Callable, optional): function that returns true if a kmer passes all
                filters. Defaults to kmer_filter_keep_all.
            min_group_size (int, optional): Smallest allowed group size. Defaults to 1.
            max_group_size (Union[int, None], optional): Largest allowed group size.  If None, then
                there is no maximum group size. Defaults to None.

        Raises:
            NotImplementedError: if kmer_source_strand and seq_coll.strands() loaded are not both
                "forward"
            ValueError: kmer_len is invalid
            ValueError: one or more group params are invalid (min_group_size, max_group_size,
                yield_first_n)
            ValueError: kmer_comparison_func is provided when kmers have not been sorted

        Returns:
            int: total_kmer_count
        """
        condition1 = self.kmer_source_strand != "forward"
        condition2 = self.seq_coll.strands_loaded() != "forward"
        if condition1 or condition2:
            raise NotImplementedError(
                f"both kmer_source_strand ({self.kmer_source_strand}) and "
                "sequence_collection.strands_loaded() must be 'forward'"
            )

        # verify that kmer_len is valid
        if kmer_len is not None and kmer_len < 1:
            raise ValueError(f"kmer_len ({kmer_len}) must be > 0")

        # verify that if kmers has not been sorted, that group arguments have not been provided
        if not self._is_sorted:
            if min_group_size != 1:
                msg = "Returning group parameters is not supported when kmers has not been"
                msg += f" sorted. min_group_size ({min_group_size}) cannot be specified. Did you"
                msg += " mean to run sort() before getting kmers?"
                raise ValueError(msg)
            if max_group_size is not None:
                msg = "Returning group parameters is not supported when kmers has not been"
                msg += f" sorted. max_group_size ({max_group_size}) cannot be specified. Did you"
                msg += " mean to run sort() before getting kmers?"
                raise ValueError(msg)

        # set kmer_comparison_func
        if self._is_sorted:
            kmer_comparison_func = get_compare_sba_kmers_func(kmer_len)
        else:
            kmer_comparison_func = compare_sba_kmers_always_less_than

        # collect values to pass to the calculation
        sba = self.seq_coll.forward_sba
        sba_strand = self.seq_coll.strands_loaded()
        kmer_start_indices = self.kmer_sba_start_indices

        # run calculation
        _, total_kmer_count = get_kmer_group_size_hist(
            sba,
            sba_strand,
            kmer_len,
            kmer_start_indices,
            kmer_comparison_func,
            kmer_filter_func,
            min_group_size,
            max_group_size,
        )

        return total_kmer_count

    def get_kmer_group_counts(
        self,
        kmer_len: Union[int, None],
        kmer_filter_func: Callable = kmer_filter_keep_all,
        min_group_size: int = 1,
        max_group_size: Union[int, None] = None,
        max_counts_bin: int = 1000000,
    ) -> tuple[np.array, int]:
        """
        A customizable function to build a histogram of group sizes for kmers passing filters.

        Examples:

        Get histogram for 50-mers
        filter = gen_kmer_length_filter_func(min_kmer_len=50)
        kmers.get_kmer_group_counts(kmer_filter_func=filter)

        Args:
            kmer_len (Union[int, None]): length of kmer.  If None, take the longest possible.
            kmer_filter_func (Callable, optional): function that returns true if a kmer passes
                all filters. Defaults to kmer_filter_keep_all.
            min_group_size (int, optional): Smallest allowed group size. Defaults to 1.
            max_group_size (Union[int, None], optional): Largest allowed group size.  If None,
                then there is no maximum group size. Defaults to None.
            max_counts_bin (int, optional): largest group_size bin in the return
                counts_by_group_size array. Group sizes that exceed this value will be placed in
                this bin. Defaults to 1000000.

        Raises:
            NotImplementedError: if kmer_source_strand and seq_coll.strands() loaded are not
            both "forward"
            ValueError: kmer_len is invalid
            ValueError: one or more group params are invalid (min_group_size, max_group_size,
                yield_first_n)
            ValueError: kmer_comparison_func is provided when kmers have not been sorted

        Returns:
            tuple[np.array, int]:
                counts_by_group_size (np.array): histogram of group sizes
                    counts_by_group_size[i] gives the number of groups of size i. Any group
                    sizes > max_counts_bin will be placed in max_counts_bin.
                total_kmer_count (int): total number of kmers
        """

        condition1 = self.kmer_source_strand != "forward"
        condition2 = self.seq_coll.strands_loaded() != "forward"
        if condition1 or condition2:
            raise NotImplementedError(
                f"both kmer_source_strand ({self.kmer_source_strand}) and "
                "sequence_collection.strands_loaded() must be 'forward'"
            )

        # verify that kmer_len is valid
        if kmer_len is not None and kmer_len < 1:
            raise ValueError(f"kmer_len ({kmer_len}) must be > 0")

        # verify that if kmers has not been sorted, that group arguments have not been provided
        if not self._is_sorted:
            if min_group_size != 1:
                msg = "Returning group parameters is not supported when kmers has not been"
                msg += f" sorted. min_group_size ({min_group_size}) cannot be specified. Did you"
                msg += " mean to run sort() before getting kmers?"
                raise ValueError(msg)
            if max_group_size is not None:
                msg = "Returning group parameters is not supported when kmers has not been"
                msg += f" sorted. max_group_size ({max_group_size}) cannot be specified. Did you"
                msg += " mean to run sort() before getting kmers?"
                raise ValueError(msg)

        # set kmer_comparison_func
        if self._is_sorted:
            kmer_comparison_func = get_compare_sba_kmers_func(kmer_len)
        else:
            raise AssertionError(f"The kmers must be sorted when calling get_kmer_group_counts")

        # collect values to pass to the calculation
        sba = self.seq_coll.forward_sba
        sba_strand = self.seq_coll.strands_loaded()
        kmer_start_indices = self.kmer_sba_start_indices

        # run calculation
        counts_by_group_size, total_kmer_count = get_kmer_group_size_hist(
            sba,
            sba_strand,
            kmer_len,
            kmer_start_indices,
            kmer_comparison_func,
            kmer_filter_func,
            min_group_size,
            max_group_size,
            max_counts_bin,
        )

        return counts_by_group_size, total_kmer_count

    def generate_get_kmer_info_func(self, one_based_seq_index: bool) -> Callable:
        """
        Generate the get_kmer_info function that is used to get kmer information from a sequence
        byte array index.

        Args:
            one_based_seq_index (bool): whether to return one-based sequence indices

        Returns:
            Callable: get_kmer_info
        """

        get_record_info_from_sba_index = self.seq_coll.generate_get_record_info_from_sba_index_func(
            one_based_seq_index
        )

        @jit
        def get_kmer_info(
            kmer_num: int,
            kmer_sba_start_indices: np.array,
            sba: np.array,
            kmer_len: Union[int, None],
            group_size_yielded: int,
            group_size_total: int,
        ) -> tuple[int, str, str, int, int, int, int]:
            """
            Given the kmer_num, return all kmer info.

            Args:
                kmer_num (int): kmer number
                kmer_sba_start_indices (np.array): sequence byte array start indices
                sba (np.array): sequence byte array
                kmer_len (Union[int, None]): length of kmer
                group_size_yielded (int): total number of kmers in the group that will be yielded
                group_size_total (int): total size of the group (including kmers not yielded)

            Raises:
                ValueError: kmer_num is invalid
                ValueError: kmer_len is invalid

            Returns:
                tuple[int, str, str, int, int, int, int]:
                    kmer_num,
                    seq_strand,
                    seq_chrom,
                    seq_start_idx,
                    kmer_len,
                    group_size_yielded,
                    group_size_total,
            """
            # verify that kmer_num is valid
            if kmer_num < 0:
                raise ValueError(f"kmer_num ({kmer_num}) cannot be less than zero")
            if kmer_num >= len(kmer_sba_start_indices):
                raise ValueError(
                    f"kmer_num ({kmer_num}) is out of bounds (num kmers = {len(kmer_sba_start_indices)})"
                )

            # get record information given the kmer's sequence byte array index
            sba_idx = kmer_sba_start_indices[kmer_num]
            seg_num, seg_sba_start_idx, seg_sba_end_idx, seq_strand, seq_chrom, seq_start_idx = (
                get_record_info_from_sba_index(sba_idx)
            )

            # if kmer_len is None, set it to the largest possible kmer
            if kmer_len is None:
                kmer_len = seg_sba_end_idx - sba_idx + 1
            # otherwise, verify that kmer_len is in-bounds
            else:
                if sba_idx + kmer_len - 1 > seg_sba_end_idx:
                    raise ValueError(
                        f"kmer_len ({kmer_len}) for kmer_num ({kmer_num}) extends beyond the end of the segment"
                    )

            return (
                kmer_num,
                seq_strand,
                seq_chrom,
                seq_start_idx,
                kmer_len,
                group_size_yielded,
                group_size_total,
            )

        return get_kmer_info

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        if self.min_kmer_len != other.min_kmer_len:
            return False

        # max_kmer_len is nullable
        if self.max_kmer_len is None and other.max_kmer_len is not None:
            return False
        elif self.max_kmer_len is not None and other.max_kmer_len is None:
            return False
        elif self.max_kmer_len != other.max_kmer_len:
            return False

        if self.kmer_source_strand != other.kmer_source_strand:
            return False
        if self.track_strands_separately != other.track_strands_separately:
            return False
        if self._is_initialized != other._is_initialized:
            return False
        if self._is_set != other._is_set:
            return False
        if self._is_sorted != other._is_sorted:
            return False

        # kmer_sba_start_indices is nullable
        if self.kmer_sba_start_indices is None and other.kmer_sba_start_indices is not None:
            return False
        elif self.kmer_sba_start_indices is not None and other.kmer_sba_start_indices is None:
            return False
        elif not np.array_equal(self.kmer_sba_start_indices, other.kmer_sba_start_indices):
            return False

        if self.seq_coll != other.seq_coll:
            return False

        # if this is reached, then they are equal
        return True

    def save(
        self,
        save_file_path: Path,
        include_sequence_collection: bool = False,
        format: str = "hdf5",
        mode: str = "w",
    ) -> None:
        """
        Save Kmers object to file.

        Args:
            save_file_path (Path): path for saved file
            include_sequence_collection (bool, optional): whether to include sequence collection.
                Defaults to False.
            format (str, optional): "hdf5" or "shelve". Defaults to "hdf5".
            mode (str, optional): mode with which to open file and save information. "w" for write
                or "a" for append. Defaults to "w".

        Raises:
            ValueError: format not recognized
        """
        if format == "hdf5":
            self._save_hdf5(save_file_path, include_sequence_collection, mode=mode)
        elif format == "shelve":
            self._save_shelve(save_file_path, include_sequence_collection)
        else:
            raise ValueError(f"format ({format}) not recognized")

    def load(
        self,
        load_file_path: Path,
        seq_coll: Union[SequenceCollection, None] = None,
        format: str = "hdf5",
    ) -> None:
        """
        Load Kmers object from saved file.

        Args:
            load_file_path (Path): path to file to load.
            seq_coll (Union[SequenceCollection, None], optional): If provided, seq_coll will be
                loaded into the kmers object rather than attempting to load from the saved file.
                Defaults to None.
            format (str, optional): "hdf5" or "shelve". Defaults to "hdf5".

        Raises:
            ValueError: format not recognized
        """
        if format == "hdf5":
            self._load_hdf5(load_file_path, seq_coll)
        elif format == "shelve":
            self._load_shelve(load_file_path, seq_coll)
        else:
            raise ValueError(f"format ({format}) not recognized")

    @staticmethod
    def _set_for_export(value, value_if_none):
        """
        Helper function that converts a None value into the appropriate indicator value if value
        equals value_if_none. This is used to help deal with the HDF5 format not handling null
        values.

        Args:
            value: value we want to pass to serialize with HDF5
            value_if_none: what to pass to HDF5 instead of value if value is None

        Returns:
            value_if_none if value is None else value
        """
        if value is None:
            return value_if_none
        else:
            return value

    @staticmethod
    def _correct_import(value, value_if_none):
        """
        Helper function that converts a value read from HDF5 to None if it equal to the indicator
        value value_if_none. This is used to help deal with the HDF5 format not handling null
        values.

        Args:
            value: value we want to pass to serialize with HDF5
            value_if_none: what to pass to HDF5 instead of value if value is None

        Returns:
            None if value == value_if_none else value
        """
        if isinstance(value, np.ndarray):
            if value.shape == (0,):
                return None
        elif value == value_if_none:
            return None
        return value

    def _save_hdf5(
        self, save_file_path: Path, include_sequence_collection: bool = False, mode: str = "w"
    ) -> None:
        """
        Save Kmers object information to HDF5 file format.

        Args:
            save_file_path (Path): path for saved file
            include_sequence_collection (bool, optional): whether to include sequence collection.
                Defaults to False.
            mode (str, optional): mode with which to open file and save information. "w" for write
                or "a" for append. Defaults to "w".
        """
        with h5py.File(save_file_path, mode) as file:
            grp = file.create_group("kmers")

            # save all class members to file.  hdf5 does not accept None values. Correct them before
            # exporting.
            empty_start_indices = np.array([], dtype=np.uint32)

            grp["min_kmer_len"] = self.min_kmer_len
            grp["max_kmer_len"] = self._set_for_export(self.max_kmer_len, 0)
            grp["kmer_source_strand"] = self.kmer_source_strand
            grp["track_strands_separately"] = self.track_strands_separately

            grp["_is_initialized"] = self._is_initialized
            grp["_is_set"] = self._is_set
            grp["_is_sorted"] = self._is_sorted
            grp["kmer_sba_start_indices"] = self._set_for_export(
                self.kmer_sba_start_indices, empty_start_indices
            )

        if include_sequence_collection:
            self.seq_coll.save(save_file_path, mode="a", format="hdf5")

    def _load_hdf5(
        self, load_file_path: Path, seq_coll: Union[SequenceCollection, None] = None
    ) -> None:
        """
        Load Kmers object information from HDF5 file format.

        Args:
            load_file_path (Path): path to file to load.
            seq_coll (Union[SequenceCollection, None], optional): If provided, seq_coll will be
                loaded into the kmers object rather than attempting to load from the saved file.
                Defaults to None.
        """
        with h5py.File(load_file_path, "r") as file:
            grp = file["kmers"]

            # read values from file
            empty_start_indices = np.array([], dtype=np.uint32)

            self.min_kmer_len = grp["min_kmer_len"][()]

            self.max_kmer_len = self._correct_import(grp["max_kmer_len"][()], 0)
            self.kmer_source_strand = grp["kmer_source_strand"][()].decode("utf-8")
            self.track_strands_separately = grp["track_strands_separately"][()]

            self._is_initialized = grp["_is_initialized"][()]
            self._is_set = grp["_is_set"][()]
            self._is_sorted = grp["_is_sorted"][()]
            self.kmer_sba_start_indices = self._correct_import(
                grp["kmer_sba_start_indices"][:], empty_start_indices
            )

        if seq_coll is not None:
            self.seq_coll = seq_coll
        else:
            self.seq_coll = SequenceCollection()
            self.seq_coll.load(load_file_path, format="hdf5")

        return

    def _save_shelve(self, save_file_path: Path, include_sequence_collection: bool = False) -> None:
        """
        Save Kmers object information to shelve file format.

        Args:
            save_file_path (Path): path for saved file
            include_sequence_collection (bool, optional): whether to include sequence collection.
                Defaults to False.
        """
        with shelve.open(save_file_path) as db:

            db["min_kmer_len"] = self.min_kmer_len
            db["max_kmer_len"] = self.max_kmer_len
            db["kmer_source_strand"] = self.kmer_source_strand
            db["track_strands_separately"] = self.track_strands_separately

            db["_is_initialized"] = self._is_initialized
            db["_is_set"] = self._is_set
            db["_is_sorted"] = self._is_sorted
            db["kmer_sba_start_indices"] = self.kmer_sba_start_indices

            db["_is_initialized"] = self._is_initialized
            db["kmer_sba_start_indices"] = self.kmer_sba_start_indices

        if include_sequence_collection:
            self.seq_coll.save(save_file_path, format="shelve")

    def _load_shelve(
        self, load_file_path: Path, seq_coll: Union[SequenceCollection, None] = None
    ) -> None:
        """
        Load Kmers object information from shelve file format.

        Args:
            load_file_path (Path): path to file to load.
            seq_coll (Union[SequenceCollection, None], optional): If provided, seq_coll will be
                loaded into the kmers object rather than attempting to load from the saved file.
                Defaults to None.
        """
        with shelve.open(load_file_path) as db:
            self.min_kmer_len = db["min_kmer_len"]
            self.max_kmer_len = db["max_kmer_len"]
            self.kmer_source_strand = db["kmer_source_strand"]
            self.track_strands_separately = db["track_strands_separately"]

            self._is_initialized = db["_is_initialized"]
            self._is_set = db["_is_set"]
            self._is_sorted = db["_is_sorted"]
            self.kmer_sba_start_indices = db["kmer_sba_start_indices"]

            self._is_initialized = db["_is_initialized"]
            self.kmer_sba_start_indices = db["kmer_sba_start_indices"]

        if seq_coll is None:
            self.seq_coll = SequenceCollection()
            self.seq_coll.load(load_file_path, format="shelve")
        else:
            self.seq_coll = seq_coll

    def get_kmer_str_no_checks(self, kmer_num: int, kmer_strand: str, kmer_len: int) -> str:
        """
        Return a string representation of kmer_num on kmer_strand with kmer_len. No checks to verify
        that arguments provided are done. Only call this function if it is known that these checks
        have already been completed (e.g. when yielded get_kmers()).

        Args:
            kmer_num (int): kmer number
            kmer_strand (str): "+" or "-"
            kmer_len (int): length of the kmer

        Raises:
            NotImplementedError: kmer_strand != "+"
            ValueError: unrecognized kmer_strand

        Returns:
            str: kmer_str
        """
        if kmer_strand == "+":
            sba = self.seq_coll.forward_sba
            sba_start_idx = self.kmer_sba_start_indices[kmer_num]
        elif kmer_strand == "-":
            raise NotImplementedError("Only implemented for kmer_strand='+'")
        else:
            raise ValueError(f"kmer_strand ({kmer_strand}) not recognized")

        return bytearray(sba[sba_start_idx : sba_start_idx + kmer_len]).decode("utf-8")

    def get_kmer_str(self, kmer_num: int, kmer_len: Union[int, None] = None) -> str:
        """
        Get the kmer_num'th kmer of kmer_len.

        Args:
            kmer_num (int): which number kmer to return (in range [0, num_kmers - 1])
            kmer_len (Union[int, None], optional): length of kmer to return. If kmer_len is None,
                return the longest possible, which is when the segment ends or the kmer_max_len
                is reached. Defaults to None.

        Raises:
            NotImplementedError: kmer_source_strand and strands_loaded must both be "forward"
            ValueError: kmer_num is invalid
            ValueError: kmer_len is invalid

        Returns:
            str: kmer
        """
        condition1 = self.kmer_source_strand != "forward"
        condition2 = self.seq_coll.strands_loaded() != "forward"
        if condition1 or condition2:
            raise NotImplementedError(
                f"both kmer_source_strand ({self.kmer_source_strand}) and "
                "sequence_collection.strands_loaded() must be 'forward'"
            )

        # verify that kmer_num is valid
        if kmer_num < 0:
            raise ValueError(f"kmer_num ({kmer_num}) cannot be less than zero")
        if kmer_num >= len(self):
            raise ValueError(f"kmer_num ({kmer_num}) is out of bounds (num kmers = {len(self)})")

        # verify that kmer_len is valid
        # TODO: consider allowing user to select a shorter or longer kmer than during sorting
        if kmer_len is not None and kmer_len < self.min_kmer_len:
            raise ValueError(
                f"kmer_len ({kmer_len}) is less than min_kmer_len ({self.min_kmer_len})"
            )
        if self.max_kmer_len is not None and kmer_len > self.max_kmer_len:
            raise ValueError(
                f"kmer_len ({kmer_len}) is greater than max_kmer_len ({self.max_kmer_len})"
            )

        sba_start_idx = self.kmer_sba_start_indices[kmer_num]
        seg_num = self.seq_coll.get_segment_num_from_sba_index(sba_start_idx)
        _, sba_seg_end_idx = self.seq_coll.get_sba_start_end_indices_for_segment(seg_num)

        if kmer_len is None:
            largest_kmer_len = sba_seg_end_idx - sba_start_idx + 1
            if self.max_kmer_len is None:
                kmer_len = largest_kmer_len
            else:
                kmer_len = min(self.max_kmer_len, largest_kmer_len)

        # verify that kmer_num is in-bounds
        if sba_start_idx + kmer_len - 1 > sba_seg_end_idx:
            raise ValueError(
                f"kmer_len ({kmer_len}) for kmer_num ({kmer_num}) extends beyond the end of the segment"
            )

        sba = self.seq_coll.forward_sba
        return bytearray(sba[sba_start_idx : sba_start_idx + kmer_len]).decode("utf-8")

    def sort(self):
        """
        Sort (in place) the kmer_sba_start_indices array by lexicographically comparing the kmers
        defined at each index.

        Raises:
            NotImplementedError: kmer_source_strand and strands_loaded must both be "forward"
        """
        condition1 = self.kmer_source_strand != "forward"
        condition2 = self.seq_coll.strands_loaded() != "forward"
        if condition1 or condition2:
            raise NotImplementedError(
                f"both kmer_source_strand ({self.kmer_source_strand}) and "
                "sequence_collection.strands_loaded() must be 'forward'"
            )

        # build the is_less_than() comparison function to be passed to quicksort
        is_less_than = self.get_is_less_than_func()

        # compile the sorting function
        quicksort_func = quicksort.make_jit_quicksort(lt=is_less_than, is_argsort=False)
        jit_sort_func = nb.njit(quicksort_func.run_quicksort)

        # sort
        jit_sort_func(self.kmer_sba_start_indices)

        self._is_sorted = True

        return

    def get_is_less_than_func(
        self, validate_kmers: bool = True, break_ties: bool = False
    ) -> Callable:
        """
        Returns a less than function that takes two integers as input and returns whether the
        kmer defined by the first index is less than the kmer defined by the second index.

        NOTE: If break_ties is True, then it will return True if the first of two equal kmers has
        a smaller sba_start_index. This is useful to gauranteeing identical output between different
        runs.  However, it comes at a significant performance cost due to additional swapping required

        Args:
            validate_kmers (bool, optional): Explicitly verify that both kmers are at least of
                min_kmer_len. Defaults to False.
            break_ties (bool, optional): if two kmers are lexicographically equivalent, break the
                tie usind the sba_start_index. Defaults to True.

        Raises:
            NotImplementedError: kmer_source_strand and strands_loaded must both be "forward"

        Returns:
            Callable: is_less_than_func
        """
        condition1 = self.kmer_source_strand != "forward"
        condition2 = self.seq_coll.strands_loaded() != "forward"
        if condition1 or condition2:
            raise NotImplementedError(
                f"both kmer_source_strand ({self.kmer_source_strand}) and "
                "sequence_collection.strands_loaded() must be 'forward'"
            )

        # assign to local variables the member variables to which is_less_than() needs access
        sba = self.seq_coll.forward_sba
        min_kmer_len = self.min_kmer_len
        max_kmer_len = self.max_kmer_len

        def is_less_than(kmer_sba_start_idx_a: int, kmer_sba_start_idx_b: int) -> bool:
            """
            Returns whether kmer_a is less than kmer_b.

            Args:
                kmer_sba_start_idx_a (int): start index in the sequence byte array for kmer_a
                kmer_sba_start_idx_b (int): start index in the sequence byte array for kmer_b

            Returns:
                bool: a_lt_b
            """
            # compare kmers
            comparison, last_kmer_index_compared = compare_sba_kmers_lexicographically(
                sba, sba, kmer_sba_start_idx_a, kmer_sba_start_idx_b, max_kmer_len=max_kmer_len
            )
            if comparison < 0:
                a_lt_b = True
            elif comparison > 0:
                a_lt_b = False
            else:
                if break_ties:
                    a_lt_b = kmer_sba_start_idx_a < kmer_sba_start_idx_b
                else:
                    a_lt_b = False

            # verify that kmer_a and kmer_b are at least of length min_kmer_len
            if validate_kmers:
                num_bases_to_check = min_kmer_len - (last_kmer_index_compared + 1)
                kmer_a_is_valid = kmer_has_required_len(
                    sba, kmer_sba_start_idx_a + last_kmer_index_compared + 1, num_bases_to_check
                )
                kmer_b_is_valid = kmer_has_required_len(
                    sba, kmer_sba_start_idx_b + last_kmer_index_compared + 1, num_bases_to_check
                )
                if not kmer_a_is_valid or not kmer_b_is_valid:
                    raise AssertionError(
                        f"kmers compared were less than min_kmer_len ({min_kmer_len}).  Was kmer_sba_start_indices initialized correctly?"
                    )

            return a_lt_b

        return is_less_than

    def to_csv(self, kmer_len, output_file_path, fields=["kmer"]):
        """
        Write all kmers to CSV file using a simple function.
        """
        pass
