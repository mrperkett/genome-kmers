import numpy as np
import pytest

from genome_kmers.sequence_collection import SequenceCollection


class TestSequenceCollection:
    # >>> ord("A"), ord("T"), ord("G"), ord("C"), ord("$")
    # (65, 84, 71, 67, 36)
    seq_list_1 = [("chr1", "ATCGAATTAG")]
    seq_list_2 = [("chr1", "ATCGAATTAG"), ("chr2", "GGATCTTGCATT"), ("chr3", "GTGATTGACCCCT")]

    def test_init_01(self):
        """
        Test sequence_list constructor with single chromosome
        """

        seq_coll = SequenceCollection(sequence_list=self.seq_list_1, strands_to_load="forward")

        assert (
            seq_coll.forward_sba
            == np.array([65, 84, 67, 71, 65, 65, 84, 84, 65, 71], dtype=np.uint8)
        ).all()
        assert seq_coll.reverse_complement_sba is None

        assert seq_coll._record_forward_sba_start_indices == np.array([0], dtype=np.uint32)
        assert seq_coll._record_reverse_complement_sba_start_indices is None

        assert seq_coll.record_names == ["chr1"]
        assert seq_coll._strands_loaded == "forward"

    def test_init_02(self):
        """
        Test sequence_list constructor with three chromosomes
        """

        seq_coll = SequenceCollection(sequence_list=self.seq_list_2, strands_to_load="forward")

        expected_forward_sba = np.zeros(37, dtype=np.uint8)
        expected_forward_sba[:10] = [65, 84, 67, 71, 65, 65, 84, 84, 65, 71]
        expected_forward_sba[10] = 36
        expected_forward_sba[11:23] = [71, 71, 65, 84, 67, 84, 84, 71, 67, 65, 84, 84]
        expected_forward_sba[23] = 36
        expected_forward_sba[24:] = [71, 84, 71, 65, 84, 84, 71, 65, 67, 67, 67, 67, 84]
        assert (seq_coll.forward_sba == expected_forward_sba).all()
        assert seq_coll.reverse_complement_sba is None

        expected_record_start_indices = np.array([0, 11, 24], dtype=np.uint32)
        assert (seq_coll._record_forward_sba_start_indices == expected_record_start_indices).all()
        assert seq_coll._record_reverse_complement_sba_start_indices is None

        assert seq_coll.record_names == ["chr1", "chr2", "chr3"]
        assert seq_coll._strands_loaded == "forward"

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
