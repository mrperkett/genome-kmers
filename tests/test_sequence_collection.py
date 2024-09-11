import os
import tempfile
from bisect import bisect_right as builtin_bisect_right
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from genome_kmers.sequence_collection import (
    SequenceCollection,
    bisect_right,
    get_forward_seq_idx,
    get_sba_start_end_indices_for_segment,
    get_segment_num_from_sba_index,
    reverse_complement_sba,
)


class TestSequenceCollection:
    """
    Contains common test data and helper functions
    """

    # example sequence_list and expected values (single chromosome)
    seq_list_1 = [("chr1", "ATCGAATTAG")]
    seq_1 = "ATCGAATTAG"
    revcomp_seq_1 = "CTAATTCGAT"
    expected_forward_sba_seq_starts_1 = np.array([0], dtype=np.uint32)
    expected_revcomp_sba_seq_starts_1 = np.array([0], dtype=np.uint32)
    expected_forward_sba_1 = np.array([ord(base) for base in seq_1], dtype=np.uint8)
    expected_revcomp_sba_1 = np.array([ord(base) for base in revcomp_seq_1], dtype=np.uint8)
    forward_record_names_1 = ["chr1"]
    revcomp_record_names_1 = ["chr1"]
    fasta_str_1 = ">chr1\nATCGAATTAG"
    revcomp_fasta_str_1 = ">chr1\nCTAATTCGAT"

    # example sequence_list and expected values (three chromosomes)
    seq_list_2 = [("chr1", "ATCGAATTAG"), ("chr2", "GGATCTTGCATT"), ("chr3", "GTGATTGACCCCT")]
    seq_2 = "ATCGAATTAG$GGATCTTGCATT$GTGATTGACCCCT"
    revcomp_seq_2 = "AGGGGTCAATCAC$AATGCAAGATCC$CTAATTCGAT"
    expected_forward_sba_seq_starts_2 = np.array([0, 11, 24], dtype=np.uint32)
    # expected_revcomp_sba_seq_starts_2 = np.array([36, 25, 12], dtype=np.uint32)
    expected_revcomp_sba_seq_starts_2 = np.array([0, 14, 27], dtype=np.uint32)
    expected_forward_sba_2 = np.array([ord(base) for base in seq_2], dtype=np.uint8)
    expected_revcomp_sba_2 = np.array([ord(base) for base in revcomp_seq_2], dtype=np.uint8)
    forward_record_names_2 = ["chr1", "chr2", "chr3"]
    revcomp_record_names_2 = ["chr3", "chr2", "chr1"]
    fasta_str_2 = ">chr1\nATCGAATTAG\n>chr2\nGGATCTTGCATT\n>chr3\nGTGATTGACCCCT"
    revcomp_fasta_str_2 = ">chr1\nCTAATTCGAT\n>chr2\nAATGCAAGATCC\n>chr3\nAGGGGTCAATCAC"

    # For testing reverse_complement
    seqs = [
        "",
        "A",
        "T",
        "G",
        "C",
        "AGCAGCCGGGT",
        "AGCAGCCGGGT$CTTAGGGAGGTGTGAGCC",
    ]
    expected_rc_seqs = [
        "",
        "T",
        "A",
        "C",
        "G",
        "ACCCGGCTGCT",
        "GGCTCACACCTCCCTAAG$ACCCGGCTGCT",
    ]
    seq_lists = [
        [("chr1", "")],
        [("chr1", "A")],
        [("chr1", "T")],
        [("chr1", "G")],
        [("chr1", "C")],
        [("chr1", "AGCAGCCGGGT")],
        [("chr1", "AGCAGCCGGGT"), ("chr2", "CTTAGGGAGGTGTGAGCC")],
    ]


class TestInitErrors(TestSequenceCollection):
    """
    Test SequenceCollection initialization errors that are independent of type of initialization
    (e.g. fasta_file_path or seq_list)
    """

    def test_init_error_01(self):
        """
        Test that you get a ValueError when attempting to initialize with both fasta_file_path and
        sequence_list
        """
        with pytest.raises(ValueError):
            SequenceCollection(
                fasta_file_path="path_to_file.fasta",
                sequence_list=self.seq_list_1,
                strands_to_load="forward",
            )

    def test_init_error_03(self):
        """
        Test that you get a ValueError if an unrecognized strands_to_load is passed
        """
        with pytest.raises(ValueError):
            SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="something_incorrect")


class TestSeqListInit(TestSequenceCollection):
    """
    Test SequenceCollection seq_list initialization
    """

    def test_forward_init_01(self):
        """
        Test sequence_list constructor with single chromosome
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="forward")

        # check sequence byte arrays
        assert np.array_equal(seq_coll.forward_sba, self.expected_forward_sba_1)
        assert seq_coll.revcomp_sba is None

        # check seq start arrays
        assert np.array_equal(
            seq_coll._forward_sba_seg_starts, self.expected_forward_sba_seq_starts_1
        )
        assert seq_coll._revcomp_sba_seg_starts is None

        # check other values that should be set
        assert seq_coll.forward_record_names == self.forward_record_names_1
        assert seq_coll.revcomp_record_names is None
        assert seq_coll._strands_loaded == "forward"

    def test_forward_init_02(self):
        """
        Test sequence_list constructor with three chromosomes
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")

        # check sequence byte arrays
        assert np.array_equal(seq_coll.forward_sba, self.expected_forward_sba_2)
        assert seq_coll.revcomp_sba is None

        # check seq start arrays
        assert np.array_equal(
            seq_coll._forward_sba_seg_starts, self.expected_forward_sba_seq_starts_2
        )
        assert seq_coll._revcomp_sba_seg_starts is None

        # check other values that should be set
        assert seq_coll.forward_record_names == self.forward_record_names_2
        assert seq_coll.revcomp_record_names is None
        assert seq_coll._strands_loaded == "forward"

    def test_revcomp_init_01(self):
        """
        Test sequence_list constructor with single chromosome
        """
        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_1, strands_to_load="reverse_complement"
        )

        # check sequence byte arrays
        assert seq_coll.forward_sba is None
        assert np.array_equal(seq_coll.revcomp_sba, self.expected_revcomp_sba_1)

        # check seq start arrays
        assert seq_coll._forward_sba_seg_starts is None
        assert np.array_equal(
            seq_coll._revcomp_sba_seg_starts, self.expected_revcomp_sba_seq_starts_1
        )

        # check other values that should be set
        assert seq_coll.revcomp_record_names == self.revcomp_record_names_1
        assert seq_coll.forward_record_names is None
        assert seq_coll._strands_loaded == "reverse_complement"

    def test_revcomp_init_02(self):
        """
        Test sequence_list constructor with three chromosomes
        """
        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_2, strands_to_load="reverse_complement"
        )

        # check sequence byte arrays
        assert seq_coll.forward_sba is None
        assert np.array_equal(seq_coll.revcomp_sba, self.expected_revcomp_sba_2)

        # check seq start arrays
        assert seq_coll._forward_sba_seg_starts is None
        assert np.array_equal(
            seq_coll._revcomp_sba_seg_starts, self.expected_revcomp_sba_seq_starts_2
        )

        # check other values that should be set
        assert seq_coll.forward_record_names is None
        assert seq_coll.revcomp_record_names == self.revcomp_record_names_2
        assert seq_coll._strands_loaded == "reverse_complement"

    def test_both_init_01(self):
        """
        Test sequence_list constructor with single chromosome
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="both")

        # check sequence byte arrays
        assert np.array_equal(seq_coll.forward_sba, self.expected_forward_sba_1)
        assert np.array_equal(seq_coll.revcomp_sba, self.expected_revcomp_sba_1)

        # check seq start arrays
        assert np.array_equal(
            seq_coll._forward_sba_seg_starts, self.expected_forward_sba_seq_starts_1
        )
        assert np.array_equal(
            seq_coll._revcomp_sba_seg_starts, self.expected_revcomp_sba_seq_starts_1
        )

        # check other values that should be set
        assert seq_coll.forward_record_names == self.forward_record_names_1
        assert seq_coll.revcomp_record_names == self.revcomp_record_names_1
        assert seq_coll._strands_loaded == "both"

    def test_both_init_02(self):
        """
        Test sequence_list constructor with three chromosomes
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="both")

        # check sequence byte arrays
        assert np.array_equal(seq_coll.forward_sba, self.expected_forward_sba_2)
        assert np.array_equal(seq_coll.revcomp_sba, self.expected_revcomp_sba_2)

        # check seq start arrays
        assert np.array_equal(
            seq_coll._forward_sba_seg_starts, self.expected_forward_sba_seq_starts_2
        )
        assert np.array_equal(
            seq_coll._revcomp_sba_seg_starts, self.expected_revcomp_sba_seq_starts_2
        )

        # check other values that should be set
        assert seq_coll.forward_record_names == self.forward_record_names_2
        assert seq_coll.revcomp_record_names == self.revcomp_record_names_2
        assert seq_coll._strands_loaded == "both"

    def test_init_error_01(self):
        """
        Non-allowed base
        """
        seq_list = [("chr1", "ATCGAATTA.")]
        with pytest.raises(ValueError):
            SequenceCollection(sequence_list=seq_list, strands_to_load="forward")

    def test_init_error_02(self):
        """
        Empty sequence
        """
        seq_list = [("chr1", "")]
        with pytest.raises(ValueError):
            SequenceCollection(sequence_list=seq_list, strands_to_load="forward")

        seq_list = [("chr1", "ATCGAATTA"), ("chr2", ""), ("chr3", "AAAATGC")]
        with pytest.raises(ValueError):
            SequenceCollection(sequence_list=seq_list, strands_to_load="forward")

    def test_init_error_03(self):
        """
        Repeated sequence record name
        """
        seq_list = [("chr1", "ATCGAATTA"), ("chr1", "AAAATGC")]
        with pytest.raises(ValueError):
            SequenceCollection(sequence_list=seq_list, strands_to_load="forward")

    def test_initialize_mapping_arrays(self):
        """
        Verify that mapping arrays initilize correctly
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="forward")

        # >>> ord("A"), ord("T"), ord("G"), ord("C"), ord("$")
        # (65, 84, 71, 67, 36)
        assert seq_coll._u1_to_uint8_mapping["A"] == 65
        assert seq_coll._u1_to_uint8_mapping["T"] == 84
        assert seq_coll._u1_to_uint8_mapping["G"] == 71
        assert seq_coll._u1_to_uint8_mapping["C"] == 67
        assert seq_coll._u1_to_uint8_mapping["$"] == 36

        assert seq_coll._uint8_to_u1_mapping[65] == "A"
        assert seq_coll._uint8_to_u1_mapping[84] == "T"
        assert seq_coll._uint8_to_u1_mapping[71] == "G"
        assert seq_coll._uint8_to_u1_mapping[67] == "C"
        assert seq_coll._uint8_to_u1_mapping[36] == "$"

        # verify that all allowed bases are in keys of _u1_to_uint8_mapping
        allowed_bases = {"A", "C", "G", "T", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N"}
        keys = set(seq_coll._u1_to_uint8_mapping.keys())
        assert allowed_bases - keys == set()


class TestFastaInit(TestSequenceCollection):
    """
    Test SequenceCollection fasta initialization
    """

    @pytest.fixture
    def mock_fasta_file_1(self, mocker):
        data = self.fasta_str_1
        mocked_data = mocker.mock_open(read_data=data)
        mocker.patch("builtins.open", mocked_data)

    @pytest.fixture
    def mock_fasta_file_2(self, mocker):
        # data = "\n".join([f">{record_name}\n{seq}" for record_name, seq in self.seq_list_2])
        data = self.fasta_str_2
        mocked_data = mocker.mock_open(read_data=data)
        mocker.patch("builtins.open", mocked_data)

    @pytest.fixture
    def mock_empty_sequence_fasta_file(self, mocker):
        data = ">chr1\nATGC\n>chr2\n\n>chr3\nATGC"
        mocked_data = mocker.mock_open(read_data=data)
        mocker.patch("builtins.open", mocked_data)

    @pytest.fixture
    def mock_fasta_file_with_illegal_base(self, mocker):
        data = ">chr1\nATGC+"
        mocked_data = mocker.mock_open(read_data=data)
        mocker.patch("builtins.open", mocked_data)

    @pytest.fixture
    def mock_fasta_file_with_repeated_record_name(self, mocker):
        data = ">chr1\nATGC\n>chr1\nATGC"
        mocked_data = mocker.mock_open(read_data=data)
        mocker.patch("builtins.open", mocked_data)

    def test_forward_init_01(self, mock_fasta_file_1):
        """
        Test sequence_list constructor with single chromosome
        """
        seq_coll = SequenceCollection(
            fasta_file_path=Path("mock_path_to_file.fa"), strands_to_load="forward"
        )

        # check sequence byte arrays
        assert np.array_equal(seq_coll.forward_sba, self.expected_forward_sba_1)
        assert seq_coll.revcomp_sba is None

        # check seq start arrays
        assert np.array_equal(
            seq_coll._forward_sba_seg_starts, self.expected_forward_sba_seq_starts_1
        )
        assert seq_coll._revcomp_sba_seg_starts is None

        # check other values that should be set
        assert seq_coll.forward_record_names == self.forward_record_names_1
        assert seq_coll.revcomp_record_names is None
        assert seq_coll._strands_loaded == "forward"

    def test_forward_init_02(self, mock_fasta_file_2):
        """
        Test sequence_list constructor with three chromosomes
        """
        seq_coll = SequenceCollection(
            fasta_file_path=Path("mock_path_to_file.fa"), strands_to_load="forward"
        )

        # check sequence byte arrays
        assert np.array_equal(seq_coll.forward_sba, self.expected_forward_sba_2)
        assert seq_coll.revcomp_sba is None

        # check seq start arrays
        assert np.array_equal(
            seq_coll._forward_sba_seg_starts, self.expected_forward_sba_seq_starts_2
        )
        assert seq_coll._revcomp_sba_seg_starts is None

        # check other values that should be set
        assert seq_coll.forward_record_names == self.forward_record_names_2
        assert seq_coll.revcomp_record_names is None
        assert seq_coll._strands_loaded == "forward"

    def test_revcomp_init_01(self, mock_fasta_file_1):
        """
        Test sequence_list constructor with single chromosome
        """
        seq_coll = SequenceCollection(
            fasta_file_path=Path("mock_path_to_file.fa"), strands_to_load="reverse_complement"
        )

        # check sequence byte arrays
        assert seq_coll.forward_sba is None
        assert np.array_equal(seq_coll.revcomp_sba, self.expected_revcomp_sba_1)

        # check seq start arrays
        assert seq_coll._forward_sba_seg_starts is None
        assert np.array_equal(
            seq_coll._revcomp_sba_seg_starts, self.expected_revcomp_sba_seq_starts_1
        )

        # check other values that should be set
        assert seq_coll.revcomp_record_names == self.revcomp_record_names_1
        assert seq_coll.forward_record_names is None
        assert seq_coll._strands_loaded == "reverse_complement"

    def test_revcomp_init_02(self, mock_fasta_file_2):
        """
        Test sequence_list constructor with three chromosomes
        """
        seq_coll = SequenceCollection(
            fasta_file_path=Path("mock_path_to_file.fa"), strands_to_load="reverse_complement"
        )

        # check sequence byte arrays
        assert seq_coll.forward_sba is None
        assert np.array_equal(seq_coll.revcomp_sba, self.expected_revcomp_sba_2)

        # check seq start arrays
        assert seq_coll._forward_sba_seg_starts is None
        assert np.array_equal(
            seq_coll._revcomp_sba_seg_starts, self.expected_revcomp_sba_seq_starts_2
        )

        # check other values that should be set
        assert seq_coll.forward_record_names is None
        assert seq_coll.revcomp_record_names == self.revcomp_record_names_2
        assert seq_coll._strands_loaded == "reverse_complement"

    def test_both_init_01(self, mock_fasta_file_1):
        """
        Test sequence_list constructor with single chromosome
        """
        seq_coll = SequenceCollection(
            fasta_file_path=Path("mock_path_to_file.fa"), strands_to_load="both"
        )

        # check sequence byte arrays
        assert np.array_equal(seq_coll.forward_sba, self.expected_forward_sba_1)
        assert np.array_equal(seq_coll.revcomp_sba, self.expected_revcomp_sba_1)

        # check seq start arrays
        assert np.array_equal(
            seq_coll._forward_sba_seg_starts, self.expected_forward_sba_seq_starts_1
        )
        assert np.array_equal(
            seq_coll._revcomp_sba_seg_starts, self.expected_revcomp_sba_seq_starts_1
        )

        # check other values that should be set
        assert seq_coll.forward_record_names == self.forward_record_names_1
        assert seq_coll.revcomp_record_names == self.revcomp_record_names_1
        assert seq_coll._strands_loaded == "both"

    def test_both_init_02(self, mock_fasta_file_2):
        """
        Test sequence_list constructor with three chromosomes
        """
        seq_coll = SequenceCollection(
            fasta_file_path=Path("mock_path_to_file.fa"), strands_to_load="both"
        )

        # check sequence byte arrays
        assert np.array_equal(seq_coll.forward_sba, self.expected_forward_sba_2)
        assert np.array_equal(seq_coll.revcomp_sba, self.expected_revcomp_sba_2)

        # check seq start arrays
        assert np.array_equal(
            seq_coll._forward_sba_seg_starts, self.expected_forward_sba_seq_starts_2
        )
        assert np.array_equal(
            seq_coll._revcomp_sba_seg_starts, self.expected_revcomp_sba_seq_starts_2
        )

        # check other values that should be set
        assert seq_coll.forward_record_names == self.forward_record_names_2
        assert seq_coll.revcomp_record_names == self.revcomp_record_names_2
        assert seq_coll._strands_loaded == "both"

    def test_init_error_01(self, mock_fasta_file_with_illegal_base):
        """
        Non-allowed base
        """
        with pytest.raises(ValueError):
            SequenceCollection(fasta_file_path=Path("mock_file_path.fa"), strands_to_load="forward")

    def test_init_error_02(self, mock_empty_sequence_fasta_file):
        """
        Empty sequence
        """
        with pytest.raises(ValueError):
            SequenceCollection(
                fasta_file_path=Path("mock_fasta_path.fa"), strands_to_load="forward"
            )

    def test_init_error_03(self, mock_fasta_file_with_repeated_record_name):
        """
        Repeated sequence record name
        """
        with pytest.raises(ValueError):
            SequenceCollection(fasta_file_path="mock_file_path.fa", strands_to_load="forward")


class TestSbaMapping(TestSequenceCollection):
    """
    Test sequence byte array mapping methods
    """

    def test_get_opposite_strand_sba_index(self):
        """
        Test that _get_opposite_strand_sba_index works as expected
        """
        assert SequenceCollection._get_opposite_strand_sba_index(0, 10) == 9
        assert SequenceCollection._get_opposite_strand_sba_index(9, 10) == 0
        assert SequenceCollection._get_opposite_strand_sba_index(3, 10) == 6

    def test_get_opposite_strand_sba_indices(self):
        """
        Test that _get_opposite_strand_sba_index works as expected
        """
        sba_indices = np.array([0, 17, 23], dtype=np.uint32)
        opposite_strand_sba_indices = SequenceCollection._get_opposite_strand_sba_indices(
            sba_indices, 30
        )
        expected_opposite_strand_sba_indices = np.array([29, 12, 6], dtype=np.uint32)
        assert (opposite_strand_sba_indices == expected_opposite_strand_sba_indices).all()

    def test_get_opposite_strand_sba_index_errors(self):
        """
        Verify that an out of bounds index will be caught
        """
        # index is out of bounds
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_index(-1, 10)
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_index(10, 10)

        # seq_len is out of bounds
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_index(0, 0)
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_index(0, -1)

    def test_get_opposite_strand_sba_indices_errors(self):
        """
        Verify that an out of bounds index will be caught
        """
        sba_indices = np.array([30, 17, 23], dtype=np.uint32)
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_indices(sba_indices, 30)

        sba_indices = np.array([0, 17, 30], dtype=np.uint32)
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_indices(sba_indices, 30)

        # seq_len is out of bounds
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_indices(sba_indices, 0)
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_indices(sba_indices, -1)

    def test_get_complement_mapping_array(self):
        """
        Does complement array look as it should?
        """
        complement_arr = SequenceCollection._get_complement_mapping_array()
        assert complement_arr[ord("A")] == ord("T")
        assert complement_arr[ord("T")] == ord("A")
        assert complement_arr[ord("G")] == ord("C")
        assert complement_arr[ord("C")] == ord("G")
        assert complement_arr[ord("$")] == ord("$")


class TestReverseComplement(TestSequenceCollection):
    """
    Test reverse complement methods
    """

    def test_reverse_complement_array_not_inplace(self):
        """
        Test that the global reverse_complement_array function works as intended for inplace=False.
        """
        for seq, expected_rc_seq in zip(self.seqs, self.expected_rc_seqs):
            expected_rc_sba = np.array([ord(base) for base in expected_rc_seq], dtype=np.uint8)
            sba = np.array([ord(base) for base in seq], dtype=np.uint8)
            complement_arr = SequenceCollection._get_complement_mapping_array()
            rc_sba = reverse_complement_sba(sba, complement_arr, inplace=False)

            # verify that rc_sba matches sba
            assert rc_sba.dtype == np.uint8
            if not np.array_equal(rc_sba, expected_rc_sba):
                raise AssertionError(
                    f"sequence byte arrays are not equal when reverse complementing seq '{seq}'"
                )
            # verify that rc_sba is not the same object as sba since inplace=False
            if len(sba) > 0:
                assert sba[0] != 255
                sba[0] = 255
                if sba[0] == rc_sba[0]:
                    raise AssertionError("Changing a value in sba also changes the value in rc_sba")

    def test_reverse_complement_array_inplace(self):
        """
        Test that the global reverse_complement_array function works as intended for inplace=False.
        """
        for seq, expected_rc_seq in zip(self.seqs, self.expected_rc_seqs):
            expected_rc_sba = np.array([ord(base) for base in expected_rc_seq], dtype=np.uint8)
            sba = np.array([ord(base) for base in seq], dtype=np.uint8)
            complement_arr = SequenceCollection._get_complement_mapping_array()
            rc_sba = reverse_complement_sba(sba, complement_arr, inplace=True)

            # verify that rc_sba matches sba
            assert rc_sba.dtype == np.uint8
            if not np.array_equal(rc_sba, expected_rc_sba):
                raise AssertionError(
                    f"sequence byte arrays are not equal when reverse complementing seq '{seq}'"
                )
            # verify that rc_sba is the same object as sba since inplace=True
            if len(sba) > 0:
                assert sba[0] != 255
                sba[0] = 255
                if sba[0] != rc_sba[0]:
                    raise AssertionError("sba and rc_sba are not the same object")

    def test_reverse_complement(self):
        """
        Test the SequenceCollection class method reverse_complement
        """
        for seq, expected_rc_seq, seq_list in zip(self.seqs, self.expected_rc_seqs, self.seq_lists):
            # only test sequences of length greater than zero
            if seq == "":
                continue

            # build expected sba for initial load and for after the reverse complement
            expected_sba = np.array([ord(base) for base in seq], dtype=np.uint8)
            expected_rc_sba = np.array([ord(base) for base in expected_rc_seq], dtype=np.uint8)
            expected_forward_record_names = [record_name for record_name, _ in seq_list]
            expected_revcomp_record_names = expected_forward_record_names.copy()
            expected_revcomp_record_names.reverse()

            # initialize a sequence collection on seq_list
            seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="forward")

            # check that everything matches what is expected before reverse complement
            assert seq_coll._strands_loaded == "forward"
            assert np.array_equal(seq_coll.forward_sba, expected_sba)
            assert seq_coll._forward_sba_seg_starts is not None
            assert seq_coll.revcomp_sba is None
            assert seq_coll._revcomp_sba_seg_starts is None
            assert seq_coll.forward_record_names == expected_forward_record_names
            assert seq_coll.revcomp_record_names is None

            seq_coll.reverse_complement()

            # check that everything matches what is expected after reverse complement
            assert seq_coll._strands_loaded == "reverse_complement"
            assert seq_coll.forward_sba is None
            assert seq_coll._forward_sba_seg_starts is None
            assert np.array_equal(seq_coll.revcomp_sba, expected_rc_sba)
            assert seq_coll._revcomp_sba_seg_starts is not None
            assert seq_coll.forward_record_names is None
            assert seq_coll.revcomp_record_names == expected_revcomp_record_names

    def test_reverse_complement_error(self):
        """
        Cannot have both strands loaded.
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="both")
        with pytest.raises(ValueError):
            seq_coll.reverse_complement()


class TestGetRecord(TestSequenceCollection):
    """
    Test get_record_num and get_record_name methods
    """

    @staticmethod
    def get_expected_segment_num(sba_start_indices, sba_idx, sba_strand):
        for i in range(len(sba_start_indices)):
            lower_bound = sba_start_indices[i]
            upper_bound = sba_start_indices[i + 1] if i != len(sba_start_indices) - 1 else 9e99
            if lower_bound <= sba_idx < upper_bound:
                return i
        raise AssertionError("Could not get expected record num.  Logic error in helper function.")

    @staticmethod
    def run_single_get_record_test(seq_list, strands_to_load, sba_strand, expected_sba_seq_starts):
        """
        Helper function to test that get_segment_num_from_sba_index() and
        get_record_name_from_sba_index() match what is expected

        Args:
            seq_list (_type_): _description_
            strands_to_load (_type_): _description_
            get_record_num_sba_strand (_type_): _description_
            expected_sba_seq_starts (_type_): _description_
        """
        # calculate the expected length of the sequence byte array (count each "$" between
        # sequences)
        total_seq_len = sum([len(seq) for _, seq in seq_list])
        assert len(seq_list) >= 1
        sba_len = total_seq_len + len(seq_list) - 1

        # initialize the sequence collection and then check if all valid indices match what is
        # expected
        seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load=strands_to_load)
        for sba_idx in range(sba_len):
            # check record_num matches what is expected
            segment_num = seq_coll.get_segment_num_from_sba_index(sba_idx, sba_strand=sba_strand)
            actual_sba_strand = sba_strand if sba_strand is not None else strands_to_load
            expected_segment_num = TestGetRecord.get_expected_segment_num(
                expected_sba_seq_starts, sba_idx, actual_sba_strand
            )
            assert segment_num == expected_segment_num

            # check record_name matches was is expected
            record_name = seq_coll.get_record_name_from_sba_index(sba_idx, sba_strand)
            if actual_sba_strand == "forward":
                expected_record_name = seq_list[segment_num][0]
            else:
                n = len(seq_list) - 1 - segment_num
                expected_record_name = seq_list[n][0]
            assert record_name == expected_record_name

    def test_get_record_from_sba_index_01(self):
        """
        Simple test case with no helper functions
        """
        # start_indices: 0, 31
        # revcomp_start_indices: 0, 21
        seq_list = [("chr1", "A" * 30), ("chr2", "T" * 20)]
        seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="forward")
        for sba_idx in range(30):
            assert seq_coll.get_record_name_from_sba_index(sba_idx) == "chr1"
        for sba_idx in range(31, 50):
            assert seq_coll.get_record_name_from_sba_index(sba_idx) == "chr2"

        seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="reverse_complement")
        for sba_idx in range(20):
            assert seq_coll.get_record_name_from_sba_index(sba_idx) == "chr2"
        for sba_idx in range(21, 50):
            assert seq_coll.get_record_name_from_sba_index(sba_idx) == "chr1"

    def test_get_record_from_sba_index_02(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_1, "forward")
            get_record_num_from_sba_index(sba_idx, sba_strand=None)
        """
        self.run_single_get_record_test(
            self.seq_list_1, "forward", None, self.expected_forward_sba_seq_starts_1
        )

    def test_get_record_from_sba_index_03(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_1, "reverse_complement")
            get_record_num_from_sba_index(sba_idx, sba_strand=None)
        """
        self.run_single_get_record_test(
            self.seq_list_1, "reverse_complement", None, self.expected_revcomp_sba_seq_starts_1
        )

    def test_get_record_from_sba_index_04(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_1, "both")
            get_record_num_from_sba_index(sba_idx, sba_strand="forward")
        """
        self.run_single_get_record_test(
            self.seq_list_1, "both", "forward", self.expected_forward_sba_seq_starts_1
        )

    def test_get_record_from_sba_index_05(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_1, "both")
            get_record_num_from_sba_index(sba_idx, sba_strand="reverse_complement")
        """
        self.run_single_get_record_test(
            self.seq_list_1, "both", "reverse_complement", self.expected_revcomp_sba_seq_starts_1
        )

    def test_get_record_from_sba_index_06(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_2, "forward")
            get_record_num_from_sba_index(sba_idx, sba_strand=None)
        """
        self.run_single_get_record_test(
            self.seq_list_2, "forward", None, self.expected_forward_sba_seq_starts_2
        )

    def test_get_record_from_sba_index_07(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_2, "reverse_complement")
            get_record_num_from_sba_index(sba_idx, sba_strand=None)
        """
        self.run_single_get_record_test(
            self.seq_list_2, "reverse_complement", None, self.expected_revcomp_sba_seq_starts_2
        )

    def test_get_record_from_sba_index_08(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_2, "both")
            get_record_num_from_sba_index(sba_idx, sba_strand="forward")
        """
        self.run_single_get_record_test(
            self.seq_list_2, "both", "forward", self.expected_forward_sba_seq_starts_2
        )

    def test_get_record_from_sba_index_09(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_2, "both")
            get_record_num_from_sba_index(sba_idx, sba_strand="reverse_complement")
        """
        self.run_single_get_record_test(
            self.seq_list_2, "both", "reverse_complement", self.expected_revcomp_sba_seq_starts_2
        )

    def test_get_record_num_from_sba_index_error_01(self):
        """
        sba_strand not specified when initialized with strands_to_load="both"
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="both")
        with pytest.raises(ValueError):
            seq_coll.get_segment_num_from_sba_index(0)

    def test_get_record_num_from_sba_index_error_02(self):
        """
        sba_strand="reverse_complement" when initialized with strands_to_load="forward"
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")
        with pytest.raises(ValueError):
            seq_coll.get_segment_num_from_sba_index(0, sba_strand="reverse_complement")

    def test_get_record_num_from_sba_index_error_03(self):
        """
        sba_strand="forward" when initialized with strands_to_load="reverse_complement"
        """
        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_2, strands_to_load="reverse_complement"
        )
        with pytest.raises(ValueError):
            seq_coll.get_segment_num_from_sba_index(0, sba_strand="forward")

    def test_get_record_num_from_sba_index_error_04(self):
        """
        unrecognized sba_strand provided
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")
        with pytest.raises(ValueError):
            seq_coll.get_segment_num_from_sba_index(0, sba_strand="unknown_value")

    def test_get_record_num_from_sba_index_error_05(self):
        """
        unrecognized sba_strand provided
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")
        with pytest.raises(IndexError):
            seq_coll.get_segment_num_from_sba_index(-1, sba_strand="forward")
        with pytest.raises(IndexError):
            seq_coll.get_segment_num_from_sba_index(37, sba_strand="forward")


class TestGetRecordLoc(TestSequenceCollection):
    """
    Test SequenceCollection functionality related to retrieve record location information

    Overview of test cases defined by expected_record_info, which uses self.seq_list_2 as input to
    SequenceCollection initialization.

    record_name:             chr1        chr2         chr3
    record_num:              [0]         [1]          [2]
    seg_num:                 [0]         [1]          [2]
    1 based fwd seq idx: 1       10 1         12 1           13
    fwd seq idx:         0        9 0         11 0           12
    sba_idx:             0        9 11        22 24          36
                         |        | |          | |           |
    seq:                 ATCGAATTAG$GGATCTTGCATT$GTGATTGACCCCT
    fwd tests:           +    +   + +  +       + +        +  +    = 9 total
    revcomp tests:       +      +    + +   +      + +    +   +    = 9 total
    revcomp:             AGGGGTCAATCAC$AATGCAAGATCC$CTAATTCGAT
                         |           | |          | |        |
    sba_idx:             0          12 14        25 27       36
    fwd seq idx:         12          0 11         0 9        0
    1 based fwd seq idx: 13          1 12         1 10       1
    seg_num:                  [0]           [1]         [2]
    record_num:               [2]           [1]         [0]
    record_name:              chr3          chr2        chr1
    """

    fwd_seg_starts = np.array([0, 11, 24], dtype=int)
    revcomp_seg_starts = np.array([0, 14, 27], dtype=int)
    Record = namedtuple(
        "Record",
        [
            "sba_idx",
            "sba_strand",
            "sba_seg_starts",
            "len_sba",
            "seg_num",
            "sba_seg_start_idx",
            "sba_seg_end_idx",
            "one_based",
            "seq_strand",
            "seq_record_name",
            "seq_start_idx",
        ],
    )
    expected_record_info = []

    # fwd test 01
    expected_record_info.append(
        Record(
            sba_idx=0,
            sba_strand="forward",
            sba_seg_starts=fwd_seg_starts,
            len_sba=37,
            seg_num=0,
            sba_seg_start_idx=0,
            sba_seg_end_idx=9,
            one_based=False,
            seq_strand="+",
            seq_record_name="chr1",
            seq_start_idx=0,
        )
    )

    # fwd test 02
    expected_record_info.append(
        Record(
            sba_idx=5,
            sba_strand="forward",
            sba_seg_starts=fwd_seg_starts,
            len_sba=37,
            seg_num=0,
            sba_seg_start_idx=0,
            sba_seg_end_idx=9,
            one_based=False,
            seq_strand="+",
            seq_record_name="chr1",
            seq_start_idx=5,
        )
    )

    # fwd test 03
    expected_record_info.append(
        Record(
            sba_idx=9,
            sba_strand="forward",
            sba_seg_starts=fwd_seg_starts,
            len_sba=37,
            seg_num=0,
            sba_seg_start_idx=0,
            sba_seg_end_idx=9,
            one_based=False,
            seq_strand="+",
            seq_record_name="chr1",
            seq_start_idx=9,
        )
    )

    # fwd test 04
    expected_record_info.append(
        Record(
            sba_idx=11,
            sba_strand="forward",
            sba_seg_starts=fwd_seg_starts,
            len_sba=37,
            seg_num=1,
            sba_seg_start_idx=11,
            sba_seg_end_idx=22,
            one_based=False,
            seq_strand="+",
            seq_record_name="chr2",
            seq_start_idx=0,
        )
    )

    # fwd test 05
    expected_record_info.append(
        Record(
            sba_idx=14,
            sba_strand="forward",
            sba_seg_starts=fwd_seg_starts,
            len_sba=37,
            seg_num=1,
            sba_seg_start_idx=11,
            sba_seg_end_idx=22,
            one_based=False,
            seq_strand="+",
            seq_record_name="chr2",
            seq_start_idx=3,
        )
    )

    # fwd test 06
    expected_record_info.append(
        Record(
            sba_idx=22,
            sba_strand="forward",
            sba_seg_starts=fwd_seg_starts,
            len_sba=37,
            seg_num=1,
            sba_seg_start_idx=11,
            sba_seg_end_idx=22,
            one_based=False,
            seq_strand="+",
            seq_record_name="chr2",
            seq_start_idx=11,
        )
    )

    # fwd test 07
    expected_record_info.append(
        Record(
            sba_idx=24,
            sba_strand="forward",
            sba_seg_starts=fwd_seg_starts,
            len_sba=37,
            seg_num=2,
            sba_seg_start_idx=24,
            sba_seg_end_idx=36,
            one_based=False,
            seq_strand="+",
            seq_record_name="chr3",
            seq_start_idx=0,
        )
    )

    # fwd test 08
    expected_record_info.append(
        Record(
            sba_idx=33,
            sba_strand="forward",
            sba_seg_starts=fwd_seg_starts,
            len_sba=37,
            seg_num=2,
            sba_seg_start_idx=24,
            sba_seg_end_idx=36,
            one_based=False,
            seq_strand="+",
            seq_record_name="chr3",
            seq_start_idx=9,
        )
    )

    # fwd test 09
    expected_record_info.append(
        Record(
            sba_idx=36,
            sba_strand="forward",
            sba_seg_starts=fwd_seg_starts,
            len_sba=37,
            seg_num=2,
            sba_seg_start_idx=24,
            sba_seg_end_idx=36,
            one_based=False,
            seq_strand="+",
            seq_record_name="chr3",
            seq_start_idx=12,
        )
    )

    # revcomp test 01
    expected_record_info.append(
        Record(
            sba_idx=0,
            sba_strand="reverse_complement",
            sba_seg_starts=revcomp_seg_starts,
            len_sba=37,
            seg_num=0,
            sba_seg_start_idx=0,
            sba_seg_end_idx=12,
            one_based=False,
            seq_strand="-",
            seq_record_name="chr3",
            seq_start_idx=12,
        )
    )

    # revcomp test 02
    expected_record_info.append(
        Record(
            sba_idx=7,
            sba_strand="reverse_complement",
            sba_seg_starts=revcomp_seg_starts,
            len_sba=37,
            seg_num=0,
            sba_seg_start_idx=0,
            sba_seg_end_idx=12,
            one_based=False,
            seq_strand="-",
            seq_record_name="chr3",
            seq_start_idx=5,
        )
    )

    # revcomp test 03
    expected_record_info.append(
        Record(
            sba_idx=12,
            sba_strand="reverse_complement",
            sba_seg_starts=revcomp_seg_starts,
            len_sba=37,
            seg_num=0,
            sba_seg_start_idx=0,
            sba_seg_end_idx=12,
            one_based=False,
            seq_strand="-",
            seq_record_name="chr3",
            seq_start_idx=0,
        )
    )

    # revcomp test 04
    expected_record_info.append(
        Record(
            sba_idx=14,
            sba_strand="reverse_complement",
            sba_seg_starts=revcomp_seg_starts,
            len_sba=37,
            seg_num=1,
            sba_seg_start_idx=14,
            sba_seg_end_idx=25,
            one_based=False,
            seq_strand="-",
            seq_record_name="chr2",
            seq_start_idx=11,
        )
    )

    # revcomp test 05
    expected_record_info.append(
        Record(
            sba_idx=18,
            sba_strand="reverse_complement",
            sba_seg_starts=revcomp_seg_starts,
            len_sba=37,
            seg_num=1,
            sba_seg_start_idx=14,
            sba_seg_end_idx=25,
            one_based=False,
            seq_strand="-",
            seq_record_name="chr2",
            seq_start_idx=7,
        )
    )

    # revcomp test 06
    expected_record_info.append(
        Record(
            sba_idx=25,
            sba_strand="reverse_complement",
            sba_seg_starts=revcomp_seg_starts,
            len_sba=37,
            seg_num=1,
            sba_seg_start_idx=14,
            sba_seg_end_idx=25,
            one_based=False,
            seq_strand="-",
            seq_record_name="chr2",
            seq_start_idx=0,
        )
    )

    # revcomp test 07
    expected_record_info.append(
        Record(
            sba_idx=27,
            sba_strand="reverse_complement",
            sba_seg_starts=revcomp_seg_starts,
            len_sba=37,
            seg_num=2,
            sba_seg_start_idx=27,
            sba_seg_end_idx=36,
            one_based=False,
            seq_strand="-",
            seq_record_name="chr1",
            seq_start_idx=9,
        )
    )

    # revcomp test 08
    expected_record_info.append(
        Record(
            sba_idx=32,
            sba_strand="reverse_complement",
            sba_seg_starts=revcomp_seg_starts,
            len_sba=37,
            seg_num=2,
            sba_seg_start_idx=27,
            sba_seg_end_idx=36,
            one_based=False,
            seq_strand="-",
            seq_record_name="chr1",
            seq_start_idx=4,
        )
    )

    # revcomp test 09
    expected_record_info.append(
        Record(
            sba_idx=36,
            sba_strand="reverse_complement",
            sba_seg_starts=revcomp_seg_starts,
            len_sba=37,
            seg_num=2,
            sba_seg_start_idx=27,
            sba_seg_end_idx=36,
            one_based=False,
            seq_strand="-",
            seq_record_name="chr1",
            seq_start_idx=0,
        )
    )

    # for each entry in expected_record_info, add corresponding entry for one_based=True
    additional_records = []
    for record in expected_record_info:
        new_record = Record(
            sba_idx=record.sba_idx,
            sba_strand=record.sba_strand,
            sba_seg_starts=record.sba_seg_starts,
            len_sba=record.len_sba,
            seg_num=record.seg_num,
            sba_seg_start_idx=record.sba_seg_start_idx,
            sba_seg_end_idx=record.sba_seg_end_idx,
            one_based=True,
            seq_strand=record.seq_strand,
            seq_record_name=record.seq_record_name,
            seq_start_idx=record.seq_start_idx + 1,
        )
        additional_records.append(new_record)
    expected_record_info.extend(additional_records)

    def test_get_record_loc_from_sba_index_01(self):
        """
        Simple test case with no helper functions
        """
        # start_indices: 0, 5
        # revcomp_start_indices: 0, 3
        # chrom: 1111 22
        # +      AAAA$GG
        # -------------
        # -      CC$TTTT
        # chrom: 22 1111
        seq_list = [("chr1", "AAAA"), ("chr2", "GG")]

        expected_forward_record_locs = {
            0: ("+", "chr1", 0),
            1: ("+", "chr1", 1),
            2: ("+", "chr1", 2),
            3: ("+", "chr1", 3),
            5: ("+", "chr2", 0),
            6: ("+", "chr2", 1),
        }

        expected_one_based_forward_record_locs = {
            0: ("+", "chr1", 1),
            1: ("+", "chr1", 2),
            2: ("+", "chr1", 3),
            3: ("+", "chr1", 4),
            5: ("+", "chr2", 1),
            6: ("+", "chr2", 2),
        }

        expected_revcomp_record_locs = {
            0: ("-", "chr2", 1),
            1: ("-", "chr2", 0),
            3: ("-", "chr1", 3),
            4: ("-", "chr1", 2),
            5: ("-", "chr1", 1),
            6: ("-", "chr1", 0),
        }

        expected_one_based_revcomp_record_locs = {
            0: ("-", "chr2", 2),
            1: ("-", "chr2", 1),
            3: ("-", "chr1", 4),
            4: ("-", "chr1", 3),
            5: ("-", "chr1", 2),
            6: ("-", "chr1", 1),
        }

        # forward, zero-based
        seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="forward")
        for sba_idx, expected_record_loc in expected_forward_record_locs.items():
            record_loc = seq_coll.get_record_loc_from_sba_index(
                sba_idx, sba_strand="forward", one_based=False
            )
            assert record_loc == expected_record_loc

        # forward, one-based
        seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="forward")
        for sba_idx, expected_record_loc in expected_one_based_forward_record_locs.items():
            record_loc = seq_coll.get_record_loc_from_sba_index(
                sba_idx, sba_strand="forward", one_based=True
            )
            assert record_loc == expected_record_loc

        # reverse complement, zero-based
        seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="reverse_complement")
        for sba_idx, expected_record_loc in expected_revcomp_record_locs.items():
            record_loc = seq_coll.get_record_loc_from_sba_index(
                sba_idx, sba_strand="reverse_complement", one_based=False
            )
            assert record_loc == expected_record_loc

        # forward, one-based
        seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="reverse_complement")
        for sba_idx, expected_record_loc in expected_one_based_revcomp_record_locs.items():
            record_loc = seq_coll.get_record_loc_from_sba_index(
                sba_idx, sba_strand="reverse_complement", one_based=True
            )
            assert record_loc == expected_record_loc

    def test_bisect_right(self):
        """
        Test bisect_right @jit implementation against the Python builtin function
        """
        seg_starts = np.array([0, 11, 24], dtype=int)
        for i in range(0, 25):
            assert bisect_right(seg_starts, i) == builtin_bisect_right(seg_starts, i)

    def test_get_forward_seq_idx(self):
        """
        Test get_forward_seq_idx @jit function following test cases outlined in the TestGetRecordLoc
        class docstring.
        """
        params = [
            ("forward", False),
            ("forward", True),
            ("reverse_complement", False),
            ("reverse_complement", True),
        ]
        for strands_to_load, one_based in params:
            # iterate through records verifying it matches expected values
            for record in self.expected_record_info:
                # skip records for a different sba_strand or one_based
                if record.sba_strand != strands_to_load or record.one_based != one_based:
                    continue

                # get_forward_seq_idx()
                fwd_seq_idx = get_forward_seq_idx(
                    sba_idx=record.sba_idx,
                    sba_strand=record.sba_strand,
                    seg_sba_start_idx=record.sba_seg_start_idx,
                    seg_sba_end_idx=record.sba_seg_end_idx,
                    one_based=record.one_based,
                )
                assert fwd_seq_idx == record.seq_start_idx

    def test_get_segment_num_from_sba_index(self):
        """
        Test get_segment_num_from_sba_index @jit function following test cases outlined in the
        TestGetRecordLoc class docstring.
        """
        params = [
            ("forward", False),
            ("forward", True),
            ("reverse_complement", False),
            ("reverse_complement", True),
        ]
        for strands_to_load, one_based in params:
            # iterate through records verifying it matches expected values
            for record in self.expected_record_info:
                # skip records for a different sba_strand or one_based
                if record.sba_strand != strands_to_load or record.one_based != one_based:
                    continue

                # get_segment_num_from_sba_index()
                seg_num = get_segment_num_from_sba_index(
                    record.sba_idx, record.sba_strand, record.sba_seg_starts
                )
                assert seg_num == record.seg_num

    def test_get_sba_start_end_indices_for_segment(self):
        """
        Test get_sba_start_end_indices_for_segment @jit function following test cases outlined in
        the TestGetRecordLoc class docstring.
        """
        params = [
            ("forward", False),
            ("forward", True),
            ("reverse_complement", False),
            ("reverse_complement", True),
        ]
        for strands_to_load, one_based in params:
            # iterate through records verifying it matches expected values
            for record in self.expected_record_info:
                # skip records for a different sba_strand or one_based
                if record.sba_strand != strands_to_load or record.one_based != one_based:
                    continue

                # get_sba_start_end_indices_for_segment()
                sba_start_index, sba_end_index = get_sba_start_end_indices_for_segment(
                    record.seg_num, record.sba_strand, record.sba_seg_starts, record.len_sba
                )
                sba_start_index == record.sba_seg_start_idx
                sba_end_index == record.sba_seg_end_idx

    def test_generate_get_record_info_from_sba_index_func(self):
        """
        Test the get_record_info_from_sba_index @jit function that is generated by the
        generate_get_record_info_from_sba_index_func member function following test cases outlined
        in the TestGetRecordLoc class docstring.
        """
        params = [
            ("forward", False),
            ("forward", True),
            ("reverse_complement", False),
            ("reverse_complement", True),
        ]
        for strands_to_load, one_based in params:
            # initialize sequence collection
            seq_coll = SequenceCollection(
                sequence_list=self.seq_list_2, strands_to_load=strands_to_load
            )

            # generate the get_record_info_from_sba_index function to test
            get_record_info_from_sba_index = seq_coll.generate_get_record_info_from_sba_index_func(
                one_based=one_based
            )

            # iterate through records verifying it matches expected values
            for record in self.expected_record_info:
                # skip records for a different sba_strand or one_based
                if record.sba_strand != strands_to_load or record.one_based != one_based:
                    continue

                # get_record_info_from_sba_index()
                (
                    seg_num,
                    seg_sba_start_idx,
                    seg_sba_end_idx,
                    seq_strand,
                    seq_record_name,
                    seq_start_idx,
                ) = get_record_info_from_sba_index(record.sba_idx)

                # verify all values meet expected
                assert seg_num == record.seg_num
                assert seg_sba_start_idx == record.sba_seg_start_idx
                assert seg_sba_end_idx == record.sba_seg_end_idx
                assert seq_strand == record.seq_strand
                assert seq_record_name == record.seq_record_name
                assert seq_start_idx == record.seq_start_idx

    def test_generate_get_record_info_from_sba_index_func_error_fwd(self):
        """
        Test requesting sba_idx out of bounds to verify an exception is raised for forward strand
        """
        # initialize sequence collection
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")

        # generate the get_record_info_from_sba_index function to test
        get_record_info_from_sba_index = seq_coll.generate_get_record_info_from_sba_index_func(
            one_based=False
        )

        # out of bounds before chr1
        with pytest.raises(ValueError):
            get_record_info_from_sba_index(-1)

        # boundary between chr1 and chr2
        with pytest.raises(ValueError):
            get_record_info_from_sba_index(10)

        # boundary between chr2 and chr3
        with pytest.raises(ValueError):
            get_record_info_from_sba_index(23)

        # out of bounds after chr3
        with pytest.raises(ValueError):
            get_record_info_from_sba_index(37)

    def test_generate_get_record_info_from_sba_index_func_error_revcomp(self):
        """
        Test requesting sba_idx out of bounds to verify an exception is raised for revcomp strand
        """
        # initialize sequence collection
        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_2, strands_to_load="reverse_complement"
        )

        # generate the get_record_info_from_sba_index function to test
        get_record_info_from_sba_index = seq_coll.generate_get_record_info_from_sba_index_func(
            one_based=False
        )

        # out of bounds before chr3
        with pytest.raises(ValueError):
            get_record_info_from_sba_index(-1)

        # boundary between chr3 and chr2
        with pytest.raises(ValueError):
            get_record_info_from_sba_index(13)

        # boundary between chr2 and chr1
        with pytest.raises(ValueError):
            get_record_info_from_sba_index(26)

        # out of bounds after chr1
        with pytest.raises(ValueError):
            get_record_info_from_sba_index(37)


class TestOtherMemberFunctions(TestSequenceCollection):
    """
    Test other member functions that don't fit into a nice category
    """

    @staticmethod
    def get_random_seq(seq_len):
        bases = ["A", "T", "G", "C"]
        seq = "".join(np.random.choice(np.array(bases, dtype="U1"), seq_len, replace=True))
        return seq

    def test_len(self):
        """ """
        np.random.seed(42)
        seq_list = []
        for i in range(5):
            chrom = f"chr{i}"
            seq = self.get_random_seq(10)
            seq_list.append((chrom, seq))

            seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="forward")
            assert len(seq_coll) == i + 1
            seq_coll.reverse_complement()
            assert len(seq_coll) == i + 1

            seq_coll = SequenceCollection(
                sequence_list=seq_list, strands_to_load="reverse_complement"
            )
            assert len(seq_coll) == i + 1
            seq_coll.reverse_complement()
            assert len(seq_coll) == i + 1

            seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="both")
            assert len(seq_coll) == i + 1

    def test_strands_loaded(self):
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")
        assert seq_coll._strands_loaded == "forward"

        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_2, strands_to_load="reverse_complement"
        )
        assert seq_coll._strands_loaded == "reverse_complement"

        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="both")
        assert seq_coll._strands_loaded == "both"

    def test_str(self):
        """
        Verify that __str__ works as expected for all loaded strand types
        """
        # seq_list_1 (single sequence)
        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="forward")
        assert str(seq_coll) == self.fasta_str_1

        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="both")
        assert str(seq_coll) == self.fasta_str_1

        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_1, strands_to_load="reverse_complement"
        )
        assert str(seq_coll) == self.revcomp_fasta_str_1

        # seq_list_2 (three sequences)
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")
        assert str(seq_coll) == self.fasta_str_2

        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="both")
        assert str(seq_coll) == self.fasta_str_2

        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_2, strands_to_load="reverse_complement"
        )
        assert str(seq_coll) == self.revcomp_fasta_str_2

    def test_iter_records(self):
        """
        Simple of iter_records. iter_records is indirectly tested through test_str as well.
        """
        # seq_list_1 (single sequence)
        # forward
        expected_records = [("chr1", 0, 9)]

        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="forward")
        records = [record for record in seq_coll.iter_records("forward")]
        assert records == expected_records

        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="both")
        records = [record for record in seq_coll.iter_records("forward")]
        assert records == expected_records

        # revcomp
        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="both")
        records = [record for record in seq_coll.iter_records("reverse_complement")]
        assert records == expected_records

        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_1, strands_to_load="reverse_complement"
        )
        records = [record for record in seq_coll.iter_records("reverse_complement")]
        assert records == expected_records

        # seq_list_2 (three sequences)
        # forward
        expected_records = [("chr1", 0, 9), ("chr2", 11, 22), ("chr3", 24, 36)]

        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")
        records = [record for record in seq_coll.iter_records("forward")]
        assert records == expected_records

        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="both")
        records = [record for record in seq_coll.iter_records("forward")]
        assert records == expected_records

        # revcomp
        expected_records = [("chr1", 27, 36), ("chr2", 14, 25), ("chr3", 0, 12)]

        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="both")
        records = [record for record in seq_coll.iter_records("reverse_complement")]
        assert records == expected_records

        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_2, strands_to_load="reverse_complement"
        )
        records = [record for record in seq_coll.iter_records("reverse_complement")]
        assert records == expected_records


class TestComparisons(TestSequenceCollection):
    @pytest.fixture
    def seq_coll_a(self):
        yield SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")

    @pytest.fixture
    def seq_coll_b(self):
        yield SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")

    def test_eq_01(self, seq_coll_a, seq_coll_b):
        """
        Equal sequence collections
        """
        assert seq_coll_a == seq_coll_b

    def test_eq_02(self, seq_coll_a, seq_coll_b):
        """
        Equal sequence collections. Differing _fasta_file_path does not impact comparison.
        """
        assert seq_coll_a == seq_coll_b
        seq_coll_b._fasta_file_path = "some_fasta_file_path"
        assert seq_coll_a == seq_coll_b

    def test_eq_03(self, seq_coll_a, seq_coll_b):
        """
        Differing forward_sba
        """
        assert seq_coll_a == seq_coll_b
        seq_coll_b.forward_sba = np.array([ord(base) for base in "ATGC"], dtype=np.uint8)
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_04(self, seq_coll_a, seq_coll_b):
        """
        Differing forward_sba
        """
        assert seq_coll_a == seq_coll_b
        seq_coll_b.forward_sba = None
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_05(self, seq_coll_a, seq_coll_b):
        """
        Differing _forward_sba_seg_starts
        """
        assert seq_coll_a == seq_coll_b
        seq_coll_b._forward_sba_seg_starts = np.array([0, 1, 2], dtype=np.uint32)
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_06(self, seq_coll_a, seq_coll_b):
        """
        Differing _forward_sba_seg_starts
        """
        assert seq_coll_a == seq_coll_b
        seq_coll_b._forward_sba_seg_starts = None
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_07(self, seq_coll_a, seq_coll_b):
        """
        Differing forward_record_names
        """
        assert seq_coll_a == seq_coll_b
        seq_coll_b.forward_record_names = ["chr1", "chr2", "chr3", "chr4"]
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_08(self, seq_coll_a, seq_coll_b):
        """
        Differing forward_record_names
        """
        assert seq_coll_a == seq_coll_b
        seq_coll_b.forward_record_names = None
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_09(self, seq_coll_a, seq_coll_b):
        """
        Differing revcomp_sba
        """
        seq_coll_a.reverse_complement()
        seq_coll_b.reverse_complement()
        assert seq_coll_a == seq_coll_b
        seq_coll_b.revcomp_sba = np.array([ord(base) for base in "ATGC"], dtype=np.uint8)
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_10(self, seq_coll_a, seq_coll_b):
        """
        Differing revcomp_sba
        """
        seq_coll_a.reverse_complement()
        seq_coll_b.reverse_complement()
        assert seq_coll_a == seq_coll_b
        seq_coll_b.revcomp_sba = None
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_11(self, seq_coll_a, seq_coll_b):
        """
        Differing _revcomp_sba_seg_starts
        """
        seq_coll_a.reverse_complement()
        seq_coll_b.reverse_complement()
        assert seq_coll_a == seq_coll_b
        seq_coll_b._revcomp_sba_seg_starts = np.array([0, 1, 2], dtype=np.uint32)
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_12(self, seq_coll_a, seq_coll_b):
        """
        Differing _revcomp_sba_seg_starts
        """
        seq_coll_a.reverse_complement()
        seq_coll_b.reverse_complement()
        assert seq_coll_a == seq_coll_b
        seq_coll_b._revcomp_sba_seg_starts = None
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_13(self, seq_coll_a, seq_coll_b):
        """
        Differing revcomp_record_names
        """
        seq_coll_a.reverse_complement()
        seq_coll_b.reverse_complement()
        assert seq_coll_a == seq_coll_b
        seq_coll_b.revcomp_record_names = ["chr1", "chr2", "chr3", "chr4"]
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_14(self, seq_coll_a, seq_coll_b):
        """
        Differing revcomp_record_names
        """
        seq_coll_a.reverse_complement()
        seq_coll_b.reverse_complement()
        assert seq_coll_a == seq_coll_b
        seq_coll_b.revcomp_record_names = None
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_15(self, seq_coll_a, seq_coll_b):
        """
        Differing _strands_loaded
        """
        assert seq_coll_a == seq_coll_b
        seq_coll_b._strands_loaded = "reverse_complement"
        assert not (seq_coll_a == seq_coll_b)

    def test_eq_16(self, seq_coll_a, seq_coll_b):
        """
        Differing _strands_loaded
        """
        assert seq_coll_a == seq_coll_b
        seq_coll_b._strands_loaded = None
        assert not (seq_coll_a == seq_coll_b)


class TestSaveLoad(TestSequenceCollection):

    @pytest.fixture
    def seq_coll_1(self):
        yield SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="forward")

    @pytest.fixture
    def seq_coll_2(self):
        yield SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")

    def test_save_load_01(self, seq_coll_1):
        """
        Test save and load for seq_coll_1 and hdf5 format
        """
        seq_coll_a = seq_coll_1
        with tempfile.NamedTemporaryFile(mode="w") as save_file:
            seq_coll_a.save(save_file_path=save_file.name, mode="w")

            seq_coll_b = SequenceCollection()
            seq_coll_b.load(save_file.name, format="hdf5")

        assert seq_coll_a == seq_coll_b

    def test_save_load_02(self, seq_coll_2):
        """
        Test save and load for seq_coll_2 and hdf5 format
        """
        seq_coll_a = seq_coll_2
        with tempfile.NamedTemporaryFile(mode="w") as save_file:
            seq_coll_a.save(save_file_path=save_file.name, mode="w")

            seq_coll_b = SequenceCollection()
            seq_coll_b.load(save_file.name, format="hdf5")

        assert seq_coll_a == seq_coll_b

    def test_save_load_03(self, seq_coll_1):
        """
        Test save and load for seq_coll_1 and shelve format
        """
        seq_coll_a = seq_coll_1
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_file_path = os.path.join(tmp_dir, "temp-shelve.db")
            seq_coll_a.save(save_file_path=save_file_path, mode="w", format="shelve")

            seq_coll_b = SequenceCollection()
            seq_coll_b.load(save_file_path, format="shelve")
            assert True

        assert seq_coll_a == seq_coll_b

    def test_save_load_04(self, seq_coll_2):
        """
        Test save and load for seq_coll_2 and shelve format
        """
        seq_coll_a = seq_coll_2
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_file_path = os.path.join(tmp_dir, "temp-shelve.db")
            seq_coll_a.save(save_file_path=save_file_path, mode="w", format="shelve")

            seq_coll_b = SequenceCollection()
            seq_coll_b.load(save_file_path, format="shelve")
            assert True

        assert seq_coll_a == seq_coll_b
