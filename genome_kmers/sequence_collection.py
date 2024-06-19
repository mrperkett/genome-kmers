from pathlib import Path

import numpy as np
from numba.core import types
from numba.typed import Dict


class SequenceCollection:
    """
    Holds all the information contained within a fasta file in a format conducive to
    kmer sorting.  Each header and its corresponding sequence is called a record.
    """

    def __init__(
        self,
        fasta_file_path: Path = None,
        sequence_list: list[tuple[str, str]] = None,
        strands_to_load: str = "forward",
    ) -> None:
        """
        Initializes a SequenceCollection object.

        Args:
            fasta_file_path (Path, optional): The path to the fasta formatted file to
                be read. Defaults to None.  Must be specified if sequence_list is not.
            sequence_list (list[tuple[str, str]], optional): List of (seq_id, seq)
                tuples defining a sequence collection. seq_id is the sequences id (i.e.
                the header in a fasta file, such as "chr1"). seq is the string sequence
                (e.g. "ACTG"). Defaults to None. Must be specified if fasta_file_path
                is not.
            strands_to_load (str, optional): which strand(s) to load into memory.  One
                of "forward", "reverse_complement", "both". Defaults to "forward".
        """
        # check provided arguments
        both_args_are_none = fasta_file_path is not None and sequence_list is not None
        neither_arg_is_none = fasta_file_path is None and sequence_list is None
        if both_args_are_none or neither_arg_is_none:
            raise ValueError(
                (
                    "Either fasta_file_path or sequence_list must be specified.  Both"
                    "cannot be specified."
                )
            )

        if strands_to_load not in ("forward", "reverse_complement", "both"):
            raise ValueError(f"strands_to_load unrecognized ({strands_to_load})")

        # https://www.bioinformatics.org/sms/iupac
        self._allowed_bases = {
            "A",
            "C",
            "G",
            "T",
            "R",
            "Y",
            "S",
            "W",
            "K",
            "M",
            "B",
            "D",
            "H",
            "V",
            "N",
            "$",
        }
        self._allowed_uint8 = {ord(base) for base in self._allowed_bases}

        # initialize arrays to map from from uint8 to character and vice versa
        self._initialize_mapping_arrays()

        # load sequence
        if fasta_file_path is not None:
            self._initialize_from_fasta()
        else:
            self._initialize_from_sequence_list(sequence_list, strands_to_load)

        return

    def __len__(self):
        """
        Equivalent to self.sequence_length()
        """
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def sequence_length(self, record_num=None, record_name=None):
        """
        Return:
            If record_num is specified, then the length of record_num.
            If record_name is specified, then the length of record_num corresponding to
            record_name
            Otherwise, the total length of all sequences
        """
        pass

    def record_names(self):
        pass

    def record_count(self):
        """
        Number of records
        """
        pass

    def sequence_info(self):
        """
        Returns a generator that yields (record_name, record_ba_start_idx,
        record_ba_end_idx)
        """
        pass

    def strands_loaded(self):
        pass

    def save_state(self, save_file_base):
        """
        Save current state to file so that it can be reloaded from disk.
        """
        pass

    def load_state(self, save_file_base):
        """
        Load state from file.
        """
        pass

    def _initialize_mapping_arrays(self):
        """
        Initialize mappings between uint8 value (the dtype stored in the sequence byte
        arrays) and the u1 value (i.e. unicode char of length 1)
        """
        self._uint8_to_u1_mapping = np.zeros(256, dtype="U1")
        self._u1_to_uint8_mapping = dict()
        self._numba_unicode_to_uint8_mapping = Dict.empty(
            key_type=types.unicode_type, value_type=types.uint8
        )
        for i in range(256):
            self._u1_to_uint8_mapping[chr(i)] = i
            self._numba_unicode_to_uint8_mapping[chr(i)] = types.uint8(i)
            self._uint8_to_u1_mapping[i] = chr(i)

        return

    def _initialize_from_fasta(self, fasta_file_path):
        pass

    def _initialize_from_sequence_list(
        self, sequence_list: list[tuple[str, str]], strands_to_load: str
    ):
        """
        Loads the sequence records from a list of (seq_id, seq) tuples.

        Args:
            sequence_list (list[tuple[str, str]]): List of (seq_id, seq)
                tuples defining a sequence collection. seq_id is the sequences id (i.e.
                the header in a fasta file, such as "chr1"). seq is the string sequence
                (e.g. "ACTG"). Defaults to None. Must be specified if fasta_file_path
                is not.
            strands_to_load (str, optional): which strand(s) to load into memory.  One
                of "forward", "reverse_complement", "both". Defaults to "forward".
        """
        self.forward_sba = None
        self._forward_sba_seq_starts = None
        self.revcomp_sba = None
        self._revcomp_sba_seq_starts = None

        # calculate the size of the sequence byte array to allocate.  Note that a '$' is placed
        # between each sequence, which accounts for the additional length
        total_seq_len = sum([len(sequence_list[i][1]) for i in range(len(sequence_list))])
        sba_length = total_seq_len + len(sequence_list) - 1

        if strands_to_load == "both":
            raise NotImplementedError()

        if strands_to_load == "forward" or strands_to_load == "both":
            self.forward_sba = np.zeros(sba_length, dtype=np.uint8)
            self._forward_sba_seq_starts = np.zeros(len(sequence_list), dtype=np.uint32)

            last_filled_idx = -1
            for i, (record_name, seq) in enumerate(sequence_list):
                start_idx = last_filled_idx + 1
                self._forward_sba_seq_starts[i] = start_idx
                self.forward_sba[start_idx : start_idx + len(seq)] = bytearray(seq, "utf-8")
                last_filled_idx = start_idx + len(seq) - 1

                # place a '$' between loaded sequences
                if i != len(sequence_list) - 1:
                    last_filled_idx += 1
                    self.forward_sba[last_filled_idx] = ord("$")

            values_in_sba = set(np.unique(self.forward_sba))
            values_not_allowed = values_in_sba - self._allowed_uint8
            if values_not_allowed != set():
                raise ValueError(
                    f"Sequence contains non-allowed characters! ({values_not_allowed})"
                )

        elif strands_to_load == "reverse_complement" or strands_to_load == "both":
            self.revcomp_sba = np.zeros(sba_length, dtype=np.uint8)
            self._revcomp_sba_seq_starts = None
            raise NotImplementedError()

        else:
            raise ValueError(f"strands_to_load not recognized ({strands_to_load})")

        # set record_names
        self.record_names = []
        for i in range(len(sequence_list)):
            record_name = sequence_list[i][0]
            self.record_names.append(record_name)

        self._strands_loaded = strands_to_load

        return

    @staticmethod
    def _reverse_complement(sba, inplace=False):
        """
        Used to reverse complement an array (e.g. self.forward_sba).  inplace is useful
        if you only need a single strand when you are done and you don't want to
        temporarily double your memory footprint.

        NOTE: will likely need nb.jit implementation

        Return:
            sba
        """
        pass

    @staticmethod
    def _get_opposite_strand_sba_index(sba_idx: int, sba_len: int) -> int:
        """
        Get the mapped sequence byte array index for the opposite strand.

        Args:
            sba_idx (int): sequence byte array index
            sba_len (int): sequence byte array length

        Returns:
            opposite_strand_sba_idx (int): the converted index
        """
        if sba_idx < 0 or sba_idx >= sba_len:
            raise ValueError(f"sba_idx ({sba_idx}) is out of bounds")
        return sba_len - 1 - sba_idx

    @staticmethod
    def _get_opposite_strand_sba_indices(sba_indices: np.array, sba_len: int) -> np.array:
        """
        Get the mapped sequence byte array indices for the opposite strand.

        Args:
            sba_indices (np.array): an array of sequence byte array indices
            sba_len (int): sequence byte arrray length

        Returns:
            opposite_strand_sba_indices: the converted indices
        """
        if (sba_indices < 0).any() or (sba_indices >= sba_len).any():
            raise ValueError("There is at least one sba index that is out of bounds")
        return sba_len - 1 - sba_indices

    def _record_name_from_ba_index(self, ba_idx, ba_strand=None):
        """
        NOTE: this may need to be sped up using nb.jit.
        """
        pass

    def _get_record_num_from_ba_index(self, ba_idx, ba_strand=None):
        """
        NOTE: this may need to be sped up using nb.jit.
        """
        pass
