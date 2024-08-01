import numpy as np
import pytest

from genome_kmers.kmers import Kmers
from genome_kmers.sequence_collection import SequenceCollection


class TestKmers:
    """
    Hold test data and functions useful to all Kmers tests
    """

    # single record
    seq_list_1 = [("chr1", "ATCGAATTAG")]
    seq_coll_1 = SequenceCollection(sequence_list=seq_list_1, strands_to_load="forward")

    # expected_kmers_array_1[min_kmer_len] = expected_kmers_array
    expected_kmers_array_1 = {}
    expected_kmers_array_1[1] = np.array([i for i in range(10)], dtype=np.uint32)
    expected_kmers_array_1[2] = np.array([i for i in range(9)], dtype=np.uint32)
    expected_kmers_array_1[10] = np.array([0], dtype=np.uint32)

    # multiple records
    seq_list_2 = [("chr1", "ATCGAATTAG"), ("chr2", "GGATCTTGCATT"), ("chr3", "GTGATTGACCCCT")]
    seq_coll_2 = SequenceCollection(sequence_list=seq_list_2, strands_to_load="forward")

    # expected_kmers_array_2[min_kmer_len] = expected_kmers_array
    expected_kmers_array_2 = {}
    idx = [i for i in range(10)] + [i for i in range(11, 11 + 12)] + [i for i in range(24, 24 + 13)]
    expected_kmers_array_2[1] = np.array(idx, dtype=np.uint32)

    idx = [i for i in range(9)] + [i for i in range(11, 11 + 11)] + [i for i in range(24, 24 + 12)]
    expected_kmers_array_2[2] = np.array(idx, dtype=np.uint32)

    idx = [i for i in range(1)] + [i for i in range(11, 11 + 3)] + [i for i in range(24, 24 + 4)]
    expected_kmers_array_2[10] = np.array(idx, dtype=np.uint32)


class TestInit(TestKmers):
    """
    Test Kmers initialization
    """

    def test_forward_single_record_01(self):
        """
        Default parameters
        """
        kmers = Kmers(self.seq_coll_1)
        assert kmers.min_kmer_len == 1
        assert kmers.max_kmer_len is None
        assert kmers.kmer_source_strand == "forward"
        assert not kmers.track_strands_separately
        assert kmers._is_initialized
        assert not kmers._is_set
        assert not kmers._is_sorted
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_1[1])

    def test_forward_single_record_02(self):
        """
        min_kmer_len=2
        """
        kmers = Kmers(
            self.seq_coll_1,
            min_kmer_len=2,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_1[2])

    def test_forward_single_record_03(self):
        """
        min_kmer_len=10 (max possible)
        """
        kmers = Kmers(
            self.seq_coll_1,
            min_kmer_len=10,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 10
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_1[10])

    def test_forward_single_record_04(self):
        """
        Verify adjusting max_kmer_len does not impact initialization (it will only impact downstream
        processing)
        max_kmer_len == min_kmer_len == 2
        """
        kmers = Kmers(
            self.seq_coll_1,
            min_kmer_len=2,
            max_kmer_len=2,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert kmers.max_kmer_len == 2
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_1[2])

    def test_forward_single_record_05(self):
        """
        Verify adjusting max_kmer_len does not impact initialization (it will only impact downstream
        processing)
            min_kmer_len = 2
            max_kmer_len = 1000 (much larger than length of sequence)
        """
        kmers = Kmers(
            self.seq_coll_1,
            min_kmer_len=2,
            max_kmer_len=1000,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert kmers.max_kmer_len == 1000
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_1[2])

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
        kmers = Kmers(self.seq_coll_2)
        assert kmers.min_kmer_len == 1
        assert kmers.max_kmer_len is None
        assert kmers.kmer_source_strand == "forward"
        assert not kmers.track_strands_separately
        assert kmers._is_initialized
        assert not kmers._is_set
        assert not kmers._is_sorted
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_2[1])

    def test_forward_multi_record_02(self):
        """
        min_kmer_len=2
        """
        kmers = Kmers(
            self.seq_coll_2,
            min_kmer_len=2,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_2[2])

    def test_forward_multi_record_03(self):
        """
        min_kmer_len=10 (max possible)
        """
        kmers = Kmers(
            self.seq_coll_2,
            min_kmer_len=10,
            max_kmer_len=None,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 10
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_2[10])

    def test_forward_multi_record_04(self):
        """
        Verify adjusting max_kmer_len does not impact initialization (it will only impact downstream
        processing)
        max_kmer_len == min_kmer_len == 2
        """
        kmers = Kmers(
            self.seq_coll_2,
            min_kmer_len=2,
            max_kmer_len=2,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert kmers.max_kmer_len == 2
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_2[2])

    def test_forward_multi_record_05(self):
        """
        Verify adjusting max_kmer_len does not impact initialization (it will only impact downstream
        processing)
            min_kmer_len = 2
            max_kmer_len = 1000 (much larger than length of sequence)
        """
        kmers = Kmers(
            self.seq_coll_2,
            min_kmer_len=2,
            max_kmer_len=1000,
            source_strand="forward",
            track_strands_separately=False,
        )
        assert kmers.min_kmer_len == 2
        assert kmers.max_kmer_len == 1000
        assert np.array_equal(kmers.kmer_sba_start_indices, self.expected_kmers_array_2[2])

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
