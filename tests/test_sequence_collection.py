import numpy as np
import pytest

from genome_kmers.sequence_collection import SequenceCollection, reverse_complement


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
    record_names_1 = ["chr1"]

    # example sequence_list and expected values (three chromosomes)
    seq_list_2 = [("chr1", "ATCGAATTAG"), ("chr2", "GGATCTTGCATT"), ("chr3", "GTGATTGACCCCT")]
    seq_2 = "ATCGAATTAG$GGATCTTGCATT$GTGATTGACCCCT"
    revcomp_seq_2 = "AGGGGTCAATCAC$AATGCAAGATCC$CTAATTCGAT"
    expected_forward_sba_seq_starts_2 = np.array([0, 11, 24], dtype=np.uint32)
    # expected_revcomp_sba_seq_starts_2 = np.array([36, 25, 12], dtype=np.uint32)
    expected_revcomp_sba_seq_starts_2 = np.array([0, 14, 27], dtype=np.uint32)
    expected_forward_sba_2 = np.array([ord(base) for base in seq_2], dtype=np.uint8)
    expected_revcomp_sba_2 = np.array([ord(base) for base in revcomp_seq_2], dtype=np.uint8)
    record_names_2 = ["chr1", "chr2", "chr3"]

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


class TestInit(TestSequenceCollection):
    """
    Test SequenceCollection initialization
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
            seq_coll._forward_sba_seq_starts, self.expected_forward_sba_seq_starts_1
        )
        assert seq_coll._revcomp_sba_seq_starts is None

        # check other values that should be set
        assert seq_coll.record_names == self.record_names_1
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
            seq_coll._forward_sba_seq_starts, self.expected_forward_sba_seq_starts_2
        )
        assert seq_coll._revcomp_sba_seq_starts is None

        # check other values that should be set
        assert seq_coll.record_names == self.record_names_2
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
        assert seq_coll._forward_sba_seq_starts is None
        assert np.array_equal(
            seq_coll._revcomp_sba_seq_starts, self.expected_revcomp_sba_seq_starts_1
        )

        # check other values that should be set
        assert seq_coll.record_names == self.record_names_1
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
        assert seq_coll._forward_sba_seq_starts is None
        assert np.array_equal(
            seq_coll._revcomp_sba_seq_starts, self.expected_revcomp_sba_seq_starts_2
        )

        # check other values that should be set
        assert seq_coll.record_names == self.record_names_2
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
            seq_coll._forward_sba_seq_starts, self.expected_forward_sba_seq_starts_1
        )
        assert np.array_equal(
            seq_coll._revcomp_sba_seq_starts, self.expected_revcomp_sba_seq_starts_1
        )

        # check other values that should be set
        assert seq_coll.record_names == self.record_names_1
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
            seq_coll._forward_sba_seq_starts, self.expected_forward_sba_seq_starts_2
        )
        assert np.array_equal(
            seq_coll._revcomp_sba_seq_starts, self.expected_revcomp_sba_seq_starts_2
        )

        # check other values that should be set
        assert seq_coll.record_names == self.record_names_2
        assert seq_coll._strands_loaded == "both"

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

    def test_init_error_02(self):
        """
        Test that you get a ValueError when attempting to initialize with neither fasta_file_path
        nor sequence_list provided
        """
        with pytest.raises(ValueError):
            SequenceCollection(strands_to_load="forward")

    def test_init_error_03(self):
        """
        Test that you get a ValueError if an unrecognized strands_to_load is passed
        """
        with pytest.raises(ValueError):
            SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="something_incorrect")

    def test_init_error_04(self):
        """
        Non-allowed base
        """
        seq_list = [("chr1", "ATCGAATTA.")]
        with pytest.raises(ValueError):
            SequenceCollection(sequence_list=seq_list, strands_to_load="forward")

    def test_init_error_05(self):
        """
        Empty sequence
        """
        seq_list = [("chr1", "")]
        with pytest.raises(ValueError):
            SequenceCollection(sequence_list=seq_list, strands_to_load="forward")

        seq_list = [("chr1", "ATCGAATTA"), ("chr2", ""), ("chr3", "AAAATGC")]
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
        sba_indices = np.array([-1, 17, 23], dtype=np.uint32)
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_indices(sba_indices, 30)

        sba_indices = np.array([0, 17, -1], dtype=np.uint32)
        with pytest.raises(ValueError):
            SequenceCollection._get_opposite_strand_sba_indices(sba_indices, 30)

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

    def test_reverse_complement_func_not_inplace(self):
        """
        Test that the global reverse_complement function works as intended for inplace=False.
        """
        for seq, expected_rc_seq in zip(self.seqs, self.expected_rc_seqs):
            expected_rc_sba = np.array([ord(base) for base in expected_rc_seq], dtype=np.uint8)
            sba = np.array([ord(base) for base in seq], dtype=np.uint8)
            complement_arr = SequenceCollection._get_complement_mapping_array()
            rc_sba = reverse_complement(sba, complement_arr, inplace=False)

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

    def test_reverse_complement_func_inplace(self):
        """
        Test that the global reverse_complement function works as intended for inplace=False.
        """
        for seq, expected_rc_seq in zip(self.seqs, self.expected_rc_seqs):
            expected_rc_sba = np.array([ord(base) for base in expected_rc_seq], dtype=np.uint8)
            sba = np.array([ord(base) for base in seq], dtype=np.uint8)
            complement_arr = SequenceCollection._get_complement_mapping_array()
            rc_sba = reverse_complement(sba, complement_arr, inplace=True)

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

            # initialize a sequence collection on seq_list
            seq_coll = SequenceCollection(sequence_list=seq_list, strands_to_load="forward")

            # check that everything matches what is expected before reverse complement
            assert seq_coll._strands_loaded == "forward"
            assert np.array_equal(seq_coll.forward_sba, expected_sba)
            assert seq_coll._forward_sba_seq_starts is not None
            assert seq_coll.revcomp_sba is None
            assert seq_coll._revcomp_sba_seq_starts is None

            seq_coll.reverse_complement()

            # check that everything matches what is expected after reverse complement
            assert seq_coll._strands_loaded == "reverse_complement"
            assert seq_coll.forward_sba is None
            assert seq_coll._forward_sba_seq_starts is None
            assert np.array_equal(seq_coll.revcomp_sba, expected_rc_sba)
            assert seq_coll._revcomp_sba_seq_starts is not None

    def test_reverse_complement_error(self):
        """
        Cannot have both strands loaded.
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="both")
        with pytest.raises(ValueError):
            seq_coll.reverse_complement()


class TestGetRecordNum(TestSequenceCollection):
    """
    Test get_record_num method
    """

    @staticmethod
    def get_expected_record_num(sba_start_indices, sba_idx):
        for i in range(len(sba_start_indices)):
            lower_bound = sba_start_indices[i]
            upper_bound = sba_start_indices[i + 1] if i != len(sba_start_indices) - 1 else 9e99
            if lower_bound <= sba_idx < upper_bound:
                return i
        raise AssertionError(f"Could not get expected record num.  Logic error in helper function.")

    @staticmethod
    def _test_get_record_num(seq_list, strands_to_load, sba_strand, expected_sba_seq_starts):
        """
        Helper function to test

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
            record_num = seq_coll.get_record_num_from_sba_index(sba_idx, sba_strand=sba_strand)
            expected_record_num = TestGetRecordNum.get_expected_record_num(
                expected_sba_seq_starts, sba_idx
            )
            assert record_num == expected_record_num

    def test_get_record_num_from_sba_index_01(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_1, "forward")
            get_record_num_from_sba_index(sba_idx, sba_strand=None)
        """
        self._test_get_record_num(
            self.seq_list_1, "forward", None, self.expected_forward_sba_seq_starts_1
        )

    def test_get_record_num_from_sba_index_02(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_1, "reverse_complement")
            get_record_num_from_sba_index(sba_idx, sba_strand=None)
        """
        self._test_get_record_num(
            self.seq_list_1, "reverse_complement", None, self.expected_revcomp_sba_seq_starts_1
        )

    def test_get_record_num_from_sba_index_03(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_1, "both")
            get_record_num_from_sba_index(sba_idx, sba_strand="forward")
        """
        self._test_get_record_num(
            self.seq_list_1, "both", "forward", self.expected_forward_sba_seq_starts_1
        )

    def test_get_record_num_from_sba_index_04(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_1, "both")
            get_record_num_from_sba_index(sba_idx, sba_strand="reverse_complement")
        """
        self._test_get_record_num(
            self.seq_list_1, "both", "reverse_complement", self.expected_revcomp_sba_seq_starts_1
        )

    def test_get_record_num_from_sba_index_05(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_2, "forward")
            get_record_num_from_sba_index(sba_idx, sba_strand=None)
        """
        self._test_get_record_num(
            self.seq_list_2, "forward", None, self.expected_forward_sba_seq_starts_2
        )

    def test_get_record_num_from_sba_index_06(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_2, "reverse_complement")
            get_record_num_from_sba_index(sba_idx, sba_strand=None)
        """
        self._test_get_record_num(
            self.seq_list_2, "reverse_complement", None, self.expected_revcomp_sba_seq_starts_2
        )

    def test_get_record_num_from_sba_index_07(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_2, "both")
            get_record_num_from_sba_index(sba_idx, sba_strand="forward")
        """
        self._test_get_record_num(
            self.seq_list_2, "both", "forward", self.expected_forward_sba_seq_starts_2
        )

    def test_get_record_num_from_sba_index_08(self):
        """
        Test all valid indices match expected record_num for:
            SequenceCollection(seq_list_2, "both")
            get_record_num_from_sba_index(sba_idx, sba_strand="reverse_complement")
        """
        self._test_get_record_num(
            self.seq_list_2, "both", "reverse_complement", self.expected_revcomp_sba_seq_starts_2
        )

    def test_get_record_num_from_sba_index_error_01(self):
        """
        sba_strand not specified when initialized with strands_to_load="both"
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="both")
        with pytest.raises(ValueError):
            seq_coll.get_record_num_from_sba_index(0)

    def test_get_record_num_from_sba_index_error_02(self):
        """
        sba_strand="reverse_complement" when initialized with strands_to_load="forward"
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")
        with pytest.raises(ValueError):
            seq_coll.get_record_num_from_sba_index(0, sba_strand="reverse_complement")

    def test_get_record_num_from_sba_index_error_03(self):
        """
        sba_strand="forward" when initialized with strands_to_load="reverse_complement"
        """
        seq_coll = SequenceCollection(
            sequence_list=self.seq_list_2, strands_to_load="reverse_complement"
        )
        with pytest.raises(ValueError):
            seq_coll.get_record_num_from_sba_index(0, sba_strand="forward")

    def test_get_record_num_from_sba_index_error_04(self):
        """
        unrecognized sba_strand provided
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")
        with pytest.raises(ValueError):
            seq_coll.get_record_num_from_sba_index(0, sba_strand="unknown_value")

    def test_get_record_num_from_sba_index_error_05(self):
        """
        unrecognized sba_strand provided
        """
        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")
        with pytest.raises(IndexError):
            seq_coll.get_record_num_from_sba_index(-1, sba_strand="forward")
        with pytest.raises(IndexError):
            seq_coll.get_record_num_from_sba_index(37, sba_strand="forward")
