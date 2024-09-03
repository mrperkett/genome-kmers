from typing import Callable, Union

import numba as nb
import numpy as np
from numba import jit
from numba.misc import quicksort

from genome_kmers.sequence_collection import SequenceCollection


@jit
def compare_sba_kmers(
    sba_a: np.array,
    sba_b: np.array,
    kmer_sba_start_idx_a: int,
    kmer_sba_start_idx_b: int,
    max_kmer_len: Union[int, None] = None,
) -> tuple[int, int]:
    """
    Lexographically compare two kmers of length kmer_len.  If kmer_len is None, the end of the
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

        pass

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

    def count_unique(self, kmer_len):
        """
        Count the number of unique kmers
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

        self._sorted = True

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
            Returns whether the kmer starting at idx_a is lexicographically less than the kmer
            starting at idx_b. Validates that both kmer_a and kmer_b are at least min_kmer_len. The
            comparison stops when max_kmer_len is reached.  If max_kmer_len is None, then the
            comparison stops upon reaching the end of a segment.

            Args:
                kmer_sba_start_idx_a (int): index in the sequence byte array at which kmer_a
                    begins
                kmer_sba_start_idx_b (int): index in the sequence byte array at which kmer_b
                    begins

            Raises:
                AssertionError: kmer_a and/or kmer_b is shorter min_kmer_len

            Returns:
                bool: kmer_a is lexicographically less than kmer_b
            """
            # initialize a_lt_b to the value for when kmers are lexicographically equal.  If
            # break_ties is defined, compare the kmer_sba_start_idx's
            if break_ties:
                a_lt_b = kmer_sba_start_idx_a < kmer_sba_start_idx_b
            else:
                a_lt_b = False

            # walk index-by-index along kmer_a and kmer_b comparing the bases at each position
            n = 0
            while True:
                # break if max_kmer_len has been reached
                if max_kmer_len is not None and n == max_kmer_len:
                    break

                # index to compare for each kmer
                idx_a = kmer_sba_start_idx_a + n
                idx_b = kmer_sba_start_idx_b + n

                # determine whether the kmer has reached the end of a segment
                idx_a_is_at_end_of_segment = (
                    True if idx_a >= len(sba) or sba[idx_a] == ord("$") else False
                )
                idx_b_is_at_end_of_segment = (
                    True if idx_b >= len(sba) or sba[idx_b] == ord("$") else False
                )

                # break if we have reached the end of a segment
                if idx_a_is_at_end_of_segment or idx_b_is_at_end_of_segment:
                    # if only idx_a has reached the end of the segment, then it is < kmer_b
                    if idx_a_is_at_end_of_segment and not idx_b_is_at_end_of_segment:
                        a_lt_b = True
                    if idx_b_is_at_end_of_segment and not idx_a_is_at_end_of_segment:
                        a_lt_b = False
                    # otherwise they are equal and the default value of a_lt_b should be used
                    break

                # check whether kmer_a < kmer_b (and vice versa)
                if sba[idx_a] < sba[idx_b]:
                    a_lt_b = True
                    break
                if sba[idx_a] > sba[idx_b]:
                    a_lt_b = False
                    break

                n += 1

            # verify that kmer_a and kmer_b are at least of length min_kmer_len
            if validate_kmers:
                for i in range(n, min_kmer_len):
                    # index to compare for each kmer
                    idx_a = kmer_sba_start_idx_a + i
                    idx_b = kmer_sba_start_idx_b + i

                    # determine whether the kmer has reached the end of a segment
                    idx_a_is_at_end_of_segment = (
                        True if idx_a >= len(sba) or sba[idx_a] == ord("$") else False
                    )
                    idx_b_is_at_end_of_segment = (
                        True if idx_b >= len(sba) or sba[idx_b] == ord("$") else False
                    )

                    # if end of segment has been reached, the kmer is invalid
                    if idx_a_is_at_end_of_segment or idx_b_is_at_end_of_segment:
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
