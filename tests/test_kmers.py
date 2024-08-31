import unittest
from typing import Union

import numpy as np
import pytest

from genome_kmers.kmers import Kmers
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

        return seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers


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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, None
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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

    def get_expected_lt_result(self, kmer_i, kmer_j, kmer_idx_i, kmer_idx_j, break_ties):
        if break_ties:
            if kmer_i == kmer_j:
                expected_result = kmer_idx_i < kmer_idx_j
            else:
                expected_result = kmer_i < kmer_j
        else:
            expected_result = kmer_i < kmer_j
        return expected_result

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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
        Test all valid parameter combinations for seq_list_1 and seq_list_2.
        """
        for seq_list in [self.seq_list_1, self.seq_list_2]:
            shortest_seq_len = min([len(seq) for _, seq in seq_list])
            min_kmer_lens = [i for i in range(1, shortest_seq_len)]
            for min_kmer_len in min_kmer_lens:
                max_kmer_lens = [j for j in range(min_kmer_len, shortest_seq_len)] + [None]
                for max_kmer_len in max_kmer_lens:
                    self.run_single_get_is_lt_func_test(
                        seq_list, min_kmer_len, max_kmer_len, break_ties=True
                    )
                    self.run_single_get_is_lt_func_test(
                        seq_list, min_kmer_len, max_kmer_len, break_ties=False
                    )

        return


class TestKmerGenerator(TestKmers):
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
        seq_coll, unsorted_kmer_indices, unsorted_kmers, sorted_kmers = self.get_expected_kmers(
            seq_list, min_kmer_len, max_kmer_len
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
