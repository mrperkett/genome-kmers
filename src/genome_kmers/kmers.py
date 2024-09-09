from typing import Callable, Generator, Union

import numba as nb
import numpy as np
from numba import jit
from numba.misc import quicksort

from genome_kmers.sequence_collection import SequenceCollection


@jit
def kmer_filter_keep_all(sba: np.array, sba_strand: str, kmer_sba_start_idx: int):
    return True


@jit
def kmer_is_valid(sba: np.array, sba_start_idx: int, min_kmer_len: int) -> bool:
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
        seq_coll: SequenceCollection,
        min_kmer_len: tuple[int, None] = None,
        max_kmer_len: tuple[int, None] = None,
        source_strand: str = "forward",
        track_strands_separately: bool = False,
        method: str = "single_pass",
    ) -> None:
        """
        Initialize Kmers

        Args:
            seq_coll (SequenceCollection): the sequence collection on which kmers are
                defined
            min_kmer_len (tuple[int, None], optional): kmers below this size are not considered.
                Defaults to None.
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
        # verify that arguments are valid
        if source_strand not in ("forward", "reverse_complement", "both"):
            raise ValueError(f"source_strand ({source_strand}) not recognized")
        if track_strands_separately not in (True, False):
            raise ValueError(
                f"track_strands_separately must be True or False, but it has value {track_strands_separately}"
            )
        if source_strand != "both" and track_strands_separately:
            raise ValueError(
                f"track_strands_separately can only be true if source_strand is 'both', but it is '{source_strand}'"
            )
        if min_kmer_len is not None and min_kmer_len < 1:
            raise ValueError(f"min_kmer_len ({min_kmer_len}) must be greater than zero")
        if max_kmer_len is not None:
            if max_kmer_len < 1:
                raise ValueError(f"max_kmer_len ({max_kmer_len}) must be greater than zero")
            if min_kmer_len is not None and max_kmer_len < min_kmer_len:
                raise ValueError(
                    f"max_kmer_len ({max_kmer_len}) is less than min_kmer_len ({min_kmer_len})"
                )

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

        # check that argument values have implemented functionality
        if track_strands_separately:
            raise NotImplementedError(
                f"This function has not been implemented for track_strands_separately = '{track_strands_separately}'"
            )
        if source_strand != "forward":
            raise NotImplementedError(
                f"This function has not been implemented for source_strand = '{source_strand}'"
            )

        # set member variables based on arguments
        self.seq_coll = seq_coll
        if min_kmer_len is None:
            self.min_kmer_len = 1
        else:
            self.min_kmer_len = min_kmer_len
        self.max_kmer_len = max_kmer_len
        self.kmer_source_strand = source_strand
        self.track_strands_separately = track_strands_separately

        self._is_initialized = False
        self._is_set = False
        self._is_sorted = False
        self.kmer_sba_start_indices = None

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

    def kmer_generator(self, kmer_len, fields=["kmer"], unique_only=False):
        """
        Defines a generator that yields tuples with the requested information about
        each kmer

        Args:
            kmer_len: size of kmer to yield

        Allowed Fields:
            kmer_num
            kmer
            reverse_complement_kmer
            canonical_kmer
            is_canonical_kmer
            record_num
            record_name
            record_seq_forward_start_idx
            record_seq_forward_end_idx
            kmer_count
            NOTE: only allowed if unique_only is True

        NOTE: I'm worried about performance, I would recommend a simple initial
        implementation and profiling
        """

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

    def save_state(self, save_file_base, include_sequence_collection):
        """
        Save current state to file so that it can be reloaded from disk.  Consider some
        sort of a check on the sequence
        file to ensure reloading from a fasta works.
        NOTE: this is a medium-to-large task to do properly with saved metadata
        """
        pass

    def load_state(self, save_file_base, sequence_collection=None):
        """
        Load state from file
        """
        pass

    def unique(self, kmer_len, inplace=False):
        """
        Discard repeated kmers keep only unique kmers.
        """
        pass

    def get_kmer(self, kmer_num: int, kmer_len: Union[int, None] = None) -> str:
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
                kmer_a_is_valid = kmer_is_valid(
                    sba, kmer_sba_start_idx_a + last_kmer_index_compared + 1, num_bases_to_check
                )
                kmer_b_is_valid = kmer_is_valid(
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
