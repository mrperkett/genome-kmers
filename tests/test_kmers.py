import itertools
import unittest
from collections import Counter
from typing import Union

import numpy as np
import pytest

from genome_kmers.kmers import (
    Kmers,
    compare_sba_kmers,
    get_compare_sba_kmers_func,
    kmer_filter_keep_all,
    kmer_nums_by_group_generator,
)
from genome_kmers.sequence_collection import SequenceCollection


class TestKmers(unittest.TestCase):
    """
    Hold test data and functions useful to all Kmers tests
    """

    def setUp(self):
        """
        Initialization to run before every test
        """

        # single record
        self.seq_list_1 = [("chr1", "ATCGAATTAG")]
        self.seq_coll_1 = SequenceCollection(
            sequence_list=self.seq_list_1, strands_to_load="forward"
        )

        # multiple records
        self.seq_list_2 = [
            ("chr1", "ATCGAATTAG"),
            ("chr2", "GGATCTTGCATT"),
            ("chr3", "GTGATTGACCCCT"),
        ]
        self.seq_coll_2 = SequenceCollection(
            sequence_list=self.seq_list_2, strands_to_load="forward"
        )

        return

    def get_parameter_combinations(self):
        """
        Helper function to build a comprehensive list of seq_list, min_kmer_len, and max_kmer_len
        to test.

        Returns:
            list: params
        """
        params = []
        for seq_list in [self.seq_list_1, self.seq_list_2]:
            shortest_seq_len = min([len(seq) for _, seq in seq_list])
            min_kmer_lens = [i for i in range(1, shortest_seq_len)]
            for min_kmer_len in min_kmer_lens:
                max_kmer_lens = [j for j in range(min_kmer_len, shortest_seq_len)] + [None]
                for max_kmer_len in max_kmer_lens:
                    params.append([seq_list, min_kmer_len, max_kmer_len])
        return params

    def get_expected_kmers(
        self, seq_list: list[tuple[str, str]], min_kmer_len: int, max_kmer_len: Union[int, None]
    ) -> tuple[SequenceCollection, np.array, list[str], list[str]]:
        """
        Helper function that generates expected output given Kmer initialization parameters

        Args:
            seq_list (list[tuple[str, str]]): from which to build the SequenceCollection
            min_kmer_len (int): minimum kmer length
            max_kmer_len (Union[int, None]): maximum kmer length used in Kmer initialization

        Returns:
            tuple[SequenceCollection, np.array, list[str], list[str]]: seq_coll,
                unsorted_kmer_indices, unsorted_kmers, sorted_kmers
        """

        # initialize a sequence collection
        seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="forward")

        # build unsorted_kmer_indices and unsorted_kmers by iterating through seq_list
        indices = []
        unsorted_kmers = []
        start_index = 0
        for seq_name, seq in seq_list:
            for i in range(len(seq) - min_kmer_len + 1):
                end = len(seq) if max_kmer_len is None else i + max_kmer_len
                kmer = seq[i:end]
                unsorted_kmers.append(kmer)
                indices.append(start_index + i)
            # +1 for "$"
            start_index += len(seq) + 1
        unsorted_kmer_indices = np.array(indices, dtype=np.uint32)

        # build sorted_kmers
        sorted_kmers = unsorted_kmers[:]
        sorted_kmers.sort()

        # build sorted_kmer_indices
        num_kmers = len(unsorted_kmers)
        sorted_kmer_numbers = [i for i in range(num_kmers)]
        sorted_kmer_numbers.sort(key=lambda idx: unsorted_kmers[idx])
        sorted_kmer_indices = [unsorted_kmer_indices[kmer_num] for kmer_num in sorted_kmer_numbers]

        return seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices


class TestInit(TestKmers):
    """
    Test Kmers initialization
    """

    def test_forward_single_record_01(self):
        """
        Default parameters
        """
        seq_list = self.seq_list_1
        min_kmer_len = 1
        max_kmer_len = None

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, None)
        )

        kmers = Kmers(seq_coll)
        assert kmers.min_kmer_len == 1
        assert kmers.max_kmer_len is None
        assert kmers.kmer_source_strand == "forward"
        assert not kmers.track_strands_separately
        assert kmers._is_initialized
        assert not kmers._is_set
        assert not kmers._is_sorted

        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_single_record_02(self):
        """
        min_kmer_len=2
        """
        seq_list = self.seq_list_1
        min_kmer_len = 2
        max_kmer_len = None

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        kmers = Kmers(
            seq_coll,
            min_kmer_len=2,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2

        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_single_record_03(self):
        """
        min_kmer_len=10 (max possible)
        """
        seq_list = self.seq_list_1
        min_kmer_len = 10
        max_kmer_len = None

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        kmers = Kmers(
            seq_coll,
            min_kmer_len=10,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 10
        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_single_record_04(self):
        """
        Verify adjusting max_kmer_len does not impact initialization (it will only impact downstream
        processing)
        max_kmer_len == min_kmer_len == 2
        """
        seq_list = self.seq_list_1
        min_kmer_len = 2
        max_kmer_len = 2

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        kmers = Kmers(
            seq_coll,
            min_kmer_len=2,
            max_kmer_len=2,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert kmers.max_kmer_len == 2

        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_single_record_05(self):
        """
        Verify adjusting max_kmer_len does not impact initialization (it will only impact downstream
        processing)
            min_kmer_len = 2
            max_kmer_len = 1000 (much larger than length of sequence)
        """
        seq_list = self.seq_list_1
        min_kmer_len = 2
        max_kmer_len = 1000

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        kmers = Kmers(
            seq_coll,
            min_kmer_len=2,
            max_kmer_len=1000,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert kmers.max_kmer_len == 1000

        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_single_record_error_01(self):
        """
        min_kmer_len out of range (min = 0)
        """
        with pytest.raises(ValueError) as e_info:
            Kmers(
                self.seq_coll_1,
                min_kmer_len=0,
                max_kmer_len=None,
                source_strand="forward",
                track_strands_separately=False,
            )
        error_msg = e_info.value.args[0]
        expected_error_msg = "min_kmer_len (0) must be greater than zero"
        assert error_msg == expected_error_msg

    def test_forward_single_record_error_02(self):
        """
        min_kmer_len out of range (min > sequence_length)
        """
        with pytest.raises(ValueError) as e_info:
            Kmers(
                self.seq_coll_1,
                min_kmer_len=100,
                max_kmer_len=None,
                source_strand="forward",
                track_strands_separately=False,
            )
        error_msg = e_info.value.args[0]
        expected_error_msg = "min_kmer_len (100) must be <= the shortest sequence length (10)"
        assert error_msg == expected_error_msg

    def test_forward_single_record_error_03(self):
        """
        max_kmer_len out of range (max = 0)
        """
        with pytest.raises(ValueError) as e_info:
            Kmers(
                self.seq_coll_1,
                min_kmer_len=None,
                max_kmer_len=0,
                source_strand="forward",
                track_strands_separately=False,
            )
        error_msg = e_info.value.args[0]
        expected_error_msg = "max_kmer_len (0) must be greater than zero"
        assert error_msg == expected_error_msg

    def test_forward_single_record_error_04(self):
        """
        max_kmer_len < min_kmer_len
        """
        with pytest.raises(ValueError) as e_info:
            Kmers(
                self.seq_coll_1,
                min_kmer_len=5,
                max_kmer_len=4,
                source_strand="forward",
                track_strands_separately=False,
            )
        error_msg = e_info.value.args[0]
        expected_error_msg = "max_kmer_len (4) is less than min_kmer_len (5)"
        assert error_msg == expected_error_msg

    def test_forward_single_record_error_05(self):
        """
        sequence_collection.strands_loaded() != source_strand
        """
        with pytest.raises(ValueError) as e_info:
            self.seq_coll_1.reverse_complement()
            Kmers(
                self.seq_coll_1,
                min_kmer_len=None,
                max_kmer_len=None,
                source_strand="forward",
                track_strands_separately=False,
            )
        error_msg = e_info.value.args[0]
        expected_error_msg = "source_strand (forward) does not match sequence_collection loaded strand (reverse_complement)"
        assert error_msg == expected_error_msg

    def test_forward_multi_record_01(self):
        """
        Default parameters
        """
        seq_list = self.seq_list_2
        min_kmer_len = 1
        max_kmer_len = None

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        kmers = Kmers(seq_coll)
        assert kmers.min_kmer_len == 1
        assert kmers.max_kmer_len is None
        assert kmers.kmer_source_strand == "forward"
        assert not kmers.track_strands_separately
        assert kmers._is_initialized
        assert not kmers._is_set
        assert not kmers._is_sorted

        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_multi_record_02(self):
        """
        min_kmer_len=2
        """
        seq_list = self.seq_list_2
        min_kmer_len = 2
        max_kmer_len = None

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        kmers = Kmers(
            seq_coll,
            min_kmer_len=2,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_multi_record_03(self):
        """
        min_kmer_len=10 (max possible)
        """
        seq_list = self.seq_list_2
        min_kmer_len = 10
        max_kmer_len = None

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        kmers = Kmers(
            seq_coll,
            min_kmer_len=10,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 10
        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_multi_record_04(self):
        """
        Verify adjusting max_kmer_len does not impact initialization (it will only impact downstream
        processing)
        max_kmer_len == min_kmer_len == 2
        """
        seq_list = self.seq_list_2
        min_kmer_len = 2
        max_kmer_len = 2

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        kmers = Kmers(
            seq_coll,
            min_kmer_len=2,
            max_kmer_len=2,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert kmers.max_kmer_len == 2
        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_multi_record_05(self):
        """
        Verify adjusting max_kmer_len does not impact initialization (it will only impact downstream
        processing)
            min_kmer_len = 2
            max_kmer_len = 1000 (much larger than length of sequence)
        """
        seq_list = self.seq_list_2
        min_kmer_len = 2
        max_kmer_len = 1000

        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        kmers = Kmers(
            seq_coll,
            min_kmer_len=2,
            max_kmer_len=1000,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert kmers.max_kmer_len == 1000

        assert np.array_equal(kmers.kmer_sba_start_indices, unsorted_kmer_indices)

    def test_forward_multi_record_error_01(self):
        """
        min_kmer_len out of range (min > sequence_length)
        """
        with pytest.raises(ValueError) as e_info:
            Kmers(
                self.seq_coll_2,
                min_kmer_len=11,
                max_kmer_len=None,
                source_strand="forward",
                track_strands_separately=False,
            )
        error_msg = e_info.value.args[0]
        expected_error_msg = "min_kmer_len (11) must be <= the shortest sequence length (10)"
        assert error_msg == expected_error_msg


class TestSort(TestKmers):
    def test_sort_01(self):
        """
        Test sorting on seq_list_1
        """
        # get expected results
        seq_list = self.seq_list_1
        min_kmer_len = 1
        max_kmer_len = None
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        # build Kmers and sort
        kmers = Kmers(
            seq_coll,
            min_kmer_len=1,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        kmers.sort()

        # verify that the sorted kmers match what is expected.  This is done by comparing the
        # longest possible kmer for each kmer_num
        assert len(kmers) == len(sorted_kmers)
        for kmer_num in range(len(kmers)):
            expected_kmer = sorted_kmers[kmer_num]
            kmer = kmers.get_kmer(kmer_num)
            assert kmer == expected_kmer

        return

    def test_sort_02(self):
        """
        Test sorting on seq_list_2
        """
        # get expected results
        seq_list = self.seq_list_2
        min_kmer_len = 1
        max_kmer_len = None
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        # build Kmers and sort
        kmers = Kmers(
            seq_coll,
            min_kmer_len=1,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        kmers.sort()

        # verify that the sorted kmers match what is expected.  This is done by comparing the
        # longest possible kmer for each kmer_num
        assert len(kmers) == len(sorted_kmers)
        for kmer_num in range(len(kmers)):
            expected_kmer = sorted_kmers[kmer_num]
            kmer = kmers.get_kmer(kmer_num)
            assert kmer == expected_kmer

        return


class TestKmerComparisons(TestKmers):

    def get_expected_lt_result(
        self,
        kmer_i: str,
        kmer_j: str,
        kmer_sba_start_idx_i: int,
        kmer_sba_start_idx_j: int,
        break_ties: bool,
    ) -> bool:
        """
        Helper function to get the expected less than comparison between two kmers strings
        (optionally breaking ties by kmer index).

        Args:
            kmer_i (str): string kmer i
            kmer_j (str): string kmer j
            kmer_sba_start_idx_i (int): kmer sequence byte array start index i
            kmer_sba_start_idx_j (int): kmer sequence byte array start index j
            break_ties (bool): whether to break kmer_i == kmer_j ties

        Returns:
            bool: kmer_i < kmer_j.  if break_ties and kmer_i == kmer_j: kmer_idx_i < kmer_idx_j
        """
        if break_ties:
            if kmer_i == kmer_j:
                expected_result = kmer_sba_start_idx_i < kmer_sba_start_idx_j
            else:
                expected_result = kmer_i < kmer_j
        else:
            expected_result = kmer_i < kmer_j
        return expected_result

    def test_get_expected_lt_result(self):
        """
        Tests verifying helper function
        """
        self.assertTrue(self.get_expected_lt_result("AT", "ATA", 0, 1, True))
        self.assertTrue(self.get_expected_lt_result("AT", "ATA", 1, 0, True))
        self.assertTrue(self.get_expected_lt_result("AT", "ATA", 0, 1, False))
        self.assertTrue(self.get_expected_lt_result("AT", "ATA", 1, 0, False))
        self.assertTrue(self.get_expected_lt_result("TTT", "TTT", 0, 1, True))
        self.assertFalse(self.get_expected_lt_result("TTT", "TTT", 1, 0, True))
        self.assertFalse(self.get_expected_lt_result("TTT", "TTT", 0, 1, False))
        self.assertFalse(self.get_expected_lt_result("TTT", "TTT", 1, 0, False))

    def get_expected_kmer_comparison(self, kmer_i: str, kmer_j: str) -> int:
        """
        Helper function to get the expected comparison between two kmer strings.

        Examples:
        kmer_i = "AAT"
        kmer_j = "AAC"
        get_expected_kmer_comparison(kmer_i, kmer_j) = 1
        get_expected_kmer_comparison(kmer_j, kmer_i) = -1
        get_expected_kmer_comparison(kmer_i[:2], kmer_j[:2]) = 0

        Args:
            kmer_i (str): string kmer i
            kmer_j (str): string kmer j

        Returns:
            int: 0 if equal, -1 if kmer_i < kmer_j, 1 if kmer_i > kmer_j
        """
        expected_comparison = None
        if kmer_i < kmer_j:
            expected_comparison = -1
        elif kmer_i == kmer_j:
            expected_comparison = 0
        else:
            expected_comparison = 1
        return expected_comparison

    def test_get_expected_kmer_comparison(self):
        """
        Tests verifying helper function
        """
        assert self.get_expected_kmer_comparison("AAT", "AAC") == 1
        assert self.get_expected_kmer_comparison("AAC", "AAT") == -1
        assert self.get_expected_kmer_comparison("AA", "AA") == 0
        assert self.get_expected_kmer_comparison("AA", "AAA") == -1
        assert self.get_expected_kmer_comparison("AAA", "AA") == 1

    def test_compare_sba_kmers_01(self):
        """
        Test compare_sba_kmers with seq_list_1 'manually'
        """
        seq_coll = self.seq_coll_1
        min_kmer_len = 1
        max_kmer_len = None

        # initialize kmers object
        kmers = Kmers(
            seq_coll,
            min_kmer_len=min_kmer_len,
            max_kmer_len=max_kmer_len,
            source_strand="forward",
            track_strands_separately=False,
        )

        # seq:                "ATCGAATTAG"
        # kmer_sba_start_idx:  0123456789
        sba = kmers.seq_coll.forward_sba

        # compare "A" and "A"
        kmer_sba_start_idx_a = 0
        kmer_sba_start_idx_b = 4
        kmer_len = 1
        comparison, last_kmer_index_compared = compare_sba_kmers(
            sba, sba, kmer_sba_start_idx_a, kmer_sba_start_idx_b, max_kmer_len=kmer_len
        )
        assert comparison == 0 and last_kmer_index_compared == 0

        # compare "AT" and "AA"
        kmer_sba_start_idx_a = 0
        kmer_sba_start_idx_b = 4
        kmer_len = 2
        comparison, last_kmer_index_compared = compare_sba_kmers(
            sba, sba, kmer_sba_start_idx_a, kmer_sba_start_idx_b, max_kmer_len=kmer_len
        )
        assert comparison == 1 and last_kmer_index_compared == 1

        # compare "AA" and "AT"
        kmer_sba_start_idx_a = 4
        kmer_sba_start_idx_b = 0
        kmer_len = 2
        comparison, last_kmer_index_compared = compare_sba_kmers(
            sba, sba, kmer_sba_start_idx_a, kmer_sba_start_idx_b, max_kmer_len=kmer_len
        )
        assert comparison == -1 and last_kmer_index_compared == 1

        # compare "GAATTAG" and "G"
        kmer_sba_start_idx_a = 3
        kmer_sba_start_idx_b = 9
        kmer_len = None
        comparison, last_kmer_index_compared = compare_sba_kmers(
            sba, sba, kmer_sba_start_idx_a, kmer_sba_start_idx_b, max_kmer_len=kmer_len
        )
        assert comparison == 1 and last_kmer_index_compared == 0

    def run_single_compare_sba_kmers_test(
        self,
        seq_list: list[tuple[str, str]],
        min_kmer_len: int,
        max_kmer_len: Union[int, None],
        source_strand: str = "forward",
    ) -> None:
        """
        Test compare_sba_kmers() by building a Kmers() object from the input parameters and
        verifying that all pairs of kmers for ever valid kmer_len match what is expected.

        Args:
            seq_list (list[tuple[str, str]]): sequence list from which to build SequenceCollection
            min_kmer_len (int): Kmers() initialization parameter
            max_kmer_len (Union[int, None]): Kmers() initialization parameter
            source_strand (str, optional): Kmers() initialization parameter. Defaults to "forward".
        """
        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        # initialize kmers object
        kmers = Kmers(
            seq_coll,
            min_kmer_len=min_kmer_len,
            max_kmer_len=max_kmer_len,
            source_strand=source_strand,
            track_strands_separately=False,
        )

        # determine the valid kmer lengths
        longest_seq_len = max([len(seq) for _, seq in seq_list])
        if max_kmer_len is None:
            valid_kmer_lens = [
                kmer_len for kmer_len in range(min_kmer_len, longest_seq_len + 1)
            ] + [None]
        else:
            valid_kmer_lens = [kmer_len for kmer_len in range(min_kmer_len, max_kmer_len + 1)]

        # there are len(unsorted_kmers) kmers.  Compare all pairs verifying that they match what
        # is expected for all valid kmer lengths
        num_kmers = len(unsorted_kmer_indices)
        sba = (
            kmers.seq_coll.forward_sba if source_strand == "forward" else kmers.seq_coll.revcomp_sba
        )
        for i in range(num_kmers):
            for j in range(num_kmers):
                for kmer_len in valid_kmer_lens:
                    # use _kmer_len to slice the kmer strings since kmer_len can be None
                    _kmer_len = longest_seq_len if kmer_len is None else kmer_len
                    kmer_i = unsorted_kmers[i][:_kmer_len]
                    kmer_j = unsorted_kmers[j][:_kmer_len]
                    kmer_sba_start_idx_i = unsorted_kmer_indices[i]
                    kmer_sba_start_idx_j = unsorted_kmer_indices[j]

                    # get the expected result of the comparison
                    expected_comparison = self.get_expected_kmer_comparison(kmer_i, kmer_j)

                    # get the result of the comparison
                    comparison, last_kmer_index_compared = compare_sba_kmers(
                        sba, sba, kmer_sba_start_idx_i, kmer_sba_start_idx_j, max_kmer_len=kmer_len
                    )

                    msg = f"i: {i} | j: {j} | kmer_len: {kmer_len} | "
                    msg += f"kmer_i: {kmer_i} | kmer_j: {kmer_j} | "
                    msg += f"comparison: {comparison} | expected_comparison: {expected_comparison}"
                    assert comparison == expected_comparison

    def test_compare_sba_kmers_comprehensive(self):
        """
        Test compare_sba_kmers() for all valid parameter combinations for seq_list_1 and
        seq_list_2.
        """
        for seq_list, min_kmer_len, max_kmer_len in self.get_parameter_combinations():
            self.run_single_compare_sba_kmers_test(seq_list, min_kmer_len, max_kmer_len, "forward")
            # TODO: add test run for "reverse_complement" after it has been implemented
        return

    def run_single_get_is_lt_func_test(
        self,
        seq_list: list[tuple[str, str]],
        min_kmer_len: int,
        max_kmer_len: Union[int, None],
        break_ties: bool,
    ):
        """
        Helper function that runs a single get_is_lt_func_test.  It tests building a less_than
        function and then testing it on all valid pairs of kmers given input parameters.

        Args:
            seq_list (list[tuple[str, str]]): from which to build the SequenceCollection
            min_kmer_len (int): minimum kmer length
            max_kmer_len (Union[int, None]): maximum kmer length used in Kmer initialization
        """
        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        # initialize kmers object
        kmers = Kmers(
            seq_coll,
            min_kmer_len=min_kmer_len,
            max_kmer_len=max_kmer_len,
            source_strand="forward",
            track_strands_separately=False,
        )

        # test is_lt_func on all pairs of values in kmer
        is_lt = kmers.get_is_less_than_func(validate_kmers=True, break_ties=break_ties)
        num_kmers = len(unsorted_kmer_indices)
        for i in range(num_kmers):
            for j in range(i + 1, num_kmers):
                # skip kmer pairs that are not of the required minimum length
                if len(unsorted_kmers[i]) < min_kmer_len or len(unsorted_kmers[j]) < min_kmer_len:
                    continue

                kmer_i = unsorted_kmers[i]
                kmer_j = unsorted_kmers[j]
                kmer_idx_i = unsorted_kmer_indices[i]
                kmer_idx_j = unsorted_kmer_indices[j]

                # check kmer_i < kmer_j
                expected_result = self.get_expected_lt_result(
                    kmer_i, kmer_j, kmer_idx_i, kmer_idx_j, break_ties
                )
                result = is_lt(kmer_idx_i, kmer_idx_j)
                msg = f"i: {i} | j: {j} | "
                msg += f"kmer_i: {kmer_i} | kmer_j: {kmer_j} | "
                msg += f"result: {result} | expected_result: {expected_result}"
                assert result == expected_result, msg

                # check kmer_j < kmer_i
                expected_result = self.get_expected_lt_result(
                    kmer_j, kmer_i, kmer_idx_j, kmer_idx_i, break_ties
                )
                result = is_lt(kmer_idx_j, kmer_idx_i)
                msg = f"i: {i} | j: {j} | "
                msg += f"kmer_i: {kmer_i} | kmer_j: {kmer_j} | "
                msg += f"result: {result} | expected_result: {expected_result}"
                assert result == expected_result, msg

    def test_get_is_lt_func(self):
        """
        Test is_lt_func() for all valid parameter combinations for seq_list_1 and
        seq_list_2.
        """
        for seq_list, min_kmer_len, max_kmer_len in self.get_parameter_combinations():
            self.run_single_get_is_lt_func_test(
                seq_list, min_kmer_len, max_kmer_len, break_ties=True
            )
            self.run_single_get_is_lt_func_test(
                seq_list, min_kmer_len, max_kmer_len, break_ties=False
            )

        return


class TestKmerGenerator(TestKmers):
    """
    Test kmer generator functions
    """

    def get_expected_group_kmers(
        self,
        sorted_kmers: list[str],
        min_group_size: int = 1,
        max_group_size: Union[int, None] = None,
        yield_first_n: Union[int, None] = None,
    ) -> tuple[list[int], int]:
        """
        Helper function that generates a list of all (kmer_nums, group_size) expected to be yielded
        by the kmer_nums_by_group_generator() given the list of kmers.  To function as intended,
        sorted_kmers must be sorted.

        Example:
        seq: ATATAGACAG
        kmer_len: 2
        unsorted_kmers: ["AT", "TA", "AT", "TA", "AG", "GA", "AC", "CA", "AG"]
        sorted_kmers: ["AC", "AG", "AG", "AT", "AT", "CA", "GA", "TA", "TA"]
        expected_kmer_nums_list = [ [0],
                                    [1, 2],
                                    [3, 4],
                                    [5],
                                    [6],
                                    [7, 8]
            ]
        expected_group_sizes = [1, 2, 2, 1, 1, 2]

        Args:
            kmers (list[str]): A list of expected kmers

        Returns:
            tuple[list[list[int]], list[int]]: expected_kmer_nums_list, expected_group_sizes
        """
        # verify sorted_kmers is sorted
        is_sorted = True
        for i in range(1, len(sorted_kmers)):
            if sorted_kmers[i] < sorted_kmers[i - 1]:
                is_sorted = False
                break
        if not is_sorted:
            raise ValueError("sorted_kmers is not sorted")

        # step through the list of kmers adding to expected_kmer_nums_list each time a new group
        # is encountered
        num_kmers = len(sorted_kmers)
        expected_kmer_nums_list = []
        group_kmer_nums = []
        for kmer_num in range(num_kmers):
            kmer = sorted_kmers[kmer_num]
            if kmer_num == 0:
                # if it's the first iteration, set prev_kmer to be the current kmer
                prev_kmer = kmer
            else:
                prev_kmer = sorted_kmers[kmer_num - 1]

            # a new group is encountered whenever kmer does not match prev_kmer
            if kmer != prev_kmer:
                expected_kmer_nums_list.append(group_kmer_nums)
                group_kmer_nums = [kmer_num]
            else:
                group_kmer_nums.append(kmer_num)

            # if we reach the end of the list of kmers, add the last group
            if kmer_num == num_kmers - 1:
                expected_kmer_nums_list.append(group_kmer_nums)

        # set expected_group_sizes
        expected_group_sizes = [len(group) for group in expected_kmer_nums_list]

        # filter by group size requirements
        for i in range(len(expected_kmer_nums_list) - 1, -1, -1):
            group_size = expected_group_sizes[i]
            passes_min_group_size = group_size >= min_group_size
            passes_max_group_size = max_group_size is None or group_size <= max_group_size
            if not passes_min_group_size or not passes_max_group_size:
                del expected_group_sizes[i]
                del expected_kmer_nums_list[i]

        # filter by yield_first_n requirements
        if yield_first_n is not None:
            for i in range(len(expected_kmer_nums_list)):
                expected_kmer_nums_list[i] = expected_kmer_nums_list[i][:yield_first_n]

        return expected_kmer_nums_list, expected_group_sizes

    def test_get_expected_group_kmers_01(self):
        """
        Verify helper function.
        """
        sorted_kmers = ["AC", "AG", "AG", "AT", "AT", "CA", "GA", "TA", "TA"]
        expected_kmer_nums_list = [[0], [1, 2], [3, 4], [5], [6], [7, 8]]
        expected_group_sizes = [1, 2, 2, 1, 1, 2]
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(sorted_kmers)
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_02(self):
        """
        Verify helper function.
        """
        # verify on self.seq_coll_2
        # sorted kmers for kmer_len = 3
        # print([kmer[:n] for kmer in sorted_kmers if len(kmer) >= n])
        sorted_kmers = [
            "AAT",
            "ACC",
            "ATC",
            "ATC",
            "ATT",
            "ATT",
            "ATT",
            "CAT",
            "CCC",
            "CCC",
            "CCT",
            "CGA",
            "CTT",
            "GAA",
            "GAC",
            "GAT",
            "GAT",
            "GCA",
            "GGA",
            "GTG",
            "TAG",
            "TCG",
            "TCT",
            "TGA",
            "TGA",
            "TGC",
            "TTA",
            "TTG",
            "TTG",
        ]
        expected_kmer_nums_list = [
            [0],
            [1],
            [2, 3],
            [4, 5, 6],
            [7],
            [8, 9],
            [10],
            [11],
            [12],
            [13],
            [14],
            [15, 16],
            [17],
            [18],
            [19],
            [20],
            [21],
            [22],
            [23, 24],
            [25],
            [26],
            [27, 28],
        ]
        expected_group_sizes = [len(group) for group in expected_kmer_nums_list]

        kmer_nums_list, group_sizes = self.get_expected_group_kmers(sorted_kmers)
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_01(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 1, max_group = 1, yield_first_n = 1
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=1, max_group_size=1, yield_first_n=1
        )
        expected_kmer_nums_list = [[4]]
        expected_group_sizes = [1]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_02(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 1, max_group = 2, yield_first_n = 1
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=1, max_group_size=2, yield_first_n=1
        )
        expected_kmer_nums_list = [[4], [5]]
        expected_group_sizes = [1, 2]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_03(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 1, max_group = 2, yield_first_n = 2
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=1, max_group_size=2, yield_first_n=2
        )
        expected_kmer_nums_list = [[4], [5, 6]]
        expected_group_sizes = [1, 2]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_04(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 1, max_group = 3, yield_first_n = None
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=1, max_group_size=3, yield_first_n=None
        )
        expected_kmer_nums_list = [[4], [5, 6], [7, 8, 9]]
        expected_group_sizes = [1, 2, 3]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_05(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 1, max_group = 3, yield_first_n = 2
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=1, max_group_size=3, yield_first_n=2
        )
        expected_kmer_nums_list = [[4], [5, 6], [7, 8]]
        expected_group_sizes = [1, 2, 3]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_06(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 1, max_group = 3, yield_first_n = 1
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=1, max_group_size=3, yield_first_n=1
        )
        expected_kmer_nums_list = [[4], [5], [7]]
        expected_group_sizes = [1, 2, 3]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_07(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 2, max_group = 3, yield_first_n = None
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=2, max_group_size=3, yield_first_n=None
        )
        expected_kmer_nums_list = [[5, 6], [7, 8, 9]]
        expected_group_sizes = [2, 3]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_08(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 2, max_group = 3, yield_first_n = 2
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=2, max_group_size=3, yield_first_n=2
        )
        expected_kmer_nums_list = [[5, 6], [7, 8]]
        expected_group_sizes = [2, 3]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_09(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 2, max_group = 3, yield_first_n = 1
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=2, max_group_size=3, yield_first_n=1
        )
        expected_kmer_nums_list = [[5], [7]]
        expected_group_sizes = [2, 3]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_10(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 4, max_group = 100, yield_first_n = None
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=4, max_group_size=100, yield_first_n=None
        )
        expected_kmer_nums_list = [[0, 1, 2, 3]]
        expected_group_sizes = [4]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_11(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 4, max_group = None, yield_first_n = None
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=4, max_group_size=None, yield_first_n=None
        )
        expected_kmer_nums_list = [[0, 1, 2, 3]]
        expected_group_sizes = [4]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def test_get_expected_group_kmers_simple_12(self):
        """
        Verify helper function.
        """
        # indices        0    1    2    3    4    5    6    7    8    9
        sorted_kmers = ["A", "A", "A", "A", "C", "G", "G", "T", "T", "T"]

        # min_group = 1, max_group = None, yield_first_n = None
        kmer_nums_list, group_sizes = self.get_expected_group_kmers(
            sorted_kmers, min_group_size=1, max_group_size=None, yield_first_n=None
        )
        expected_kmer_nums_list = [[0, 1, 2, 3], [4], [5, 6], [7, 8, 9]]
        expected_group_sizes = [4, 1, 2, 3]
        self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
        self.assertListEqual(group_sizes, expected_group_sizes)

    def run_single_get_kmer_test(
        self, seq_list: list[tuple[str, str]], min_kmer_len: int, max_kmer_len: Union[int, None]
    ):
        """
        Helper function that runs a single get_kmer_test.  It tests getting all kmers of valid
        length given the Kmers() object.

        Args:
            seq_list (list[tuple[str, str]]): from which to build the SequenceCollection
            min_kmer_len (int): minimum kmer length
            max_kmer_len (Union[int, None]): maximum kmer length used in Kmer initialization
        """
        # get expected results based on input arguments
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        # initialize kmers object
        kmers = Kmers(
            seq_coll,
            min_kmer_len=min_kmer_len,
            max_kmer_len=max_kmer_len,
            source_strand="forward",
            track_strands_separately=False,
        )

        # determine valid kmer_lens to test
        shortest_seq_len = min([len(seq) for _, seq in seq_list])
        if max_kmer_len is not None:
            largest_kmer_len = min(shortest_seq_len, max_kmer_len)
        else:
            largest_kmer_len = shortest_seq_len
        kmer_lens = range(min_kmer_len, largest_kmer_len)

        # test each kmer of valid length
        for kmer_num, full_kmer in enumerate(unsorted_kmers):
            for kmer_len in kmer_lens:
                if len(full_kmer) < kmer_len:
                    continue
                expected_kmer = full_kmer[:kmer_len]
                kmer = kmers.get_kmer(kmer_num, kmer_len)
                assert kmer == expected_kmer

        return

    def test_get_kmer(self):
        """
        Test all valid parameter combinations for seq_list_1 and seq_list_2
        """
        for seq_list in [self.seq_list_1, self.seq_list_2]:
            shortest_seq_len = min([len(seq) for _, seq in seq_list])
            min_kmer_lens = [i for i in range(1, shortest_seq_len)]
            for min_kmer_len in min_kmer_lens:
                max_kmer_lens = [j for j in range(min_kmer_len, shortest_seq_len)] + [None]
                for max_kmer_len in max_kmer_lens:
                    self.run_single_get_kmer_test(seq_list, min_kmer_len, max_kmer_len)

        return

    def run_single_kmer_nums_by_group_generator_test(
        self, seq_list: list[tuple[str, str]], kmer_len: int
    ) -> None:
        """
        Run a single kmer_nums_by_group_generator test for a given seq_list and kmer_len.  Note
        that kmers will be exactly of kmer_len and this function does not test variable length
        kmers.

        NOTE: The parameters tested in this function are currently hard coded so that they
        work in a reasonable amount of time on self.seq_list_2.  This could be adjusted if
        necessary in the future.

        Args:
            seq_list (list[tuple[str, str]]): sequence list for SequenceCollection
            kmer_len (int): length of kmer to test
        """
        # define the parameters to test the generator agains - they all impact group yielding
        min_group_size_list = [1, 2, 3, 4]
        max_group_size_list = [1, 2, 3, 4, 7, None]
        yield_first_n_list = [1, 2, 3, 4, 7, None]
        group_params = []
        for min_group_size in min_group_size_list:
            for max_group_size in max_group_size_list:
                # NOTE: max_group_size cannot be less than min_group_size
                if max_group_size is not None and max_group_size < min_group_size:
                    continue
                for yield_first_n in yield_first_n_list:
                    group_params.append([min_group_size, max_group_size, yield_first_n])

        # get expected results based on input arguments
        min_kmer_len = kmer_len
        max_kmer_len = kmer_len
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers, sorted_kmer_indices = (
            self.get_expected_kmers(seq_list, min_kmer_len, max_kmer_len)
        )

        # initialize kmers object
        kmers = Kmers(
            seq_coll,
            min_kmer_len=min_kmer_len,
            max_kmer_len=max_kmer_len,
            source_strand="forward",
            track_strands_separately=False,
        )

        # sort kmers
        kmers.sort()

        # initialize the kmer_nums_by_group generator.  Set filter to keep all
        sba = seq_coll.forward_sba
        sba_strand = kmers.seq_coll.strands_loaded()
        kmer_start_indices = kmers.kmer_sba_start_indices
        kmer_comparison_func = get_compare_sba_kmers_func(kmer_len)
        kmer_filter_func = kmer_filter_keep_all

        # for each set of group yielding parameters, test the the generator yields as expected
        for min_group_size, max_group_size, yield_first_n in group_params:
            # initialize generator
            generator = kmer_nums_by_group_generator(
                sba=sba,
                sba_strand=sba_strand,
                kmer_start_indices=kmer_start_indices,
                kmer_comparison_func=kmer_comparison_func,
                kmer_filter_func=kmer_filter_func,
                min_group_size=min_group_size,
                max_group_size=max_group_size,
                yield_first_n=yield_first_n,
            )

            # collect all items in the generator
            kmer_nums_list = []
            group_sizes = []
            for kmer_nums, group_size in generator:
                kmer_nums_list.append(kmer_nums)
                group_sizes.append(group_size)

            # check that everything matches what is expected
            expected_kmer_nums_list, expected_group_sizes = self.get_expected_group_kmers(
                sorted_kmers, min_group_size, max_group_size, yield_first_n
            )
            self.assertListEqual(kmer_nums_list, expected_kmer_nums_list)
            self.assertListEqual(group_sizes, expected_group_sizes)

    def test_kmer_nums_by_group_generator_comprehensive(self):
        """
        Comprehensive set of tests for kmer_nums_by_group_generator over seq_list_2 for varied
        kmer lengths (variable kmer length is NOT tested).

        NOTE: This test could be expanded to greater parameters at the cost of run time.
        """
        # the parameters over which to vary the test
        seq_lists = [self.seq_list_2]
        kmer_len_list = [1, 2, 3, 4, 8]

        # run the tests
        for seq_list in seq_lists:
            for kmer_len in kmer_len_list:
                self.run_single_kmer_nums_by_group_generator_test(seq_list, kmer_len)

    def test_kmer_nums_by_group_generator_01(self):
        """
        Test a single kmer_nums_by_group_generator instance
        """
        seq_list = [("chr1", "ATCGAATTAG")]
        kmer_len = 1
        self.run_single_kmer_nums_by_group_generator_test(seq_list, kmer_len)
