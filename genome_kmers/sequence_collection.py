from pathlib import Path

import numpy as np
from numba import jit
from numba.core import types
from numba.typed import Dict


@jit
def reverse_complement(sba: np.array, complement_mapping_arr: np.array, inplace=False) -> np.array:
    """
    Reverse complement sequence byte array (sba) using the uint8 to uint8 mapping array
    (complement_mapping_arr).  This function uses numba.jit for performance.

    Args:
        sba (np.array): sequence byte array (dtype=uint8)
        complement_mapping_arr (np.array): maps from sequence byte array value (uint8) to
            complement sequence byte array value (uint8)
        inplace (bool, optional): whether to perform in place or return a newly created array.
            Defaults to False.

    Returns:
        np.array: reverse complemented sequence byte array
    """
    if inplace:
        for idx in range((len(sba) + 1) // 2):
            rc_idx = len(sba) - 1 - idx
            front_byte = sba[idx]
            back_byte = sba[rc_idx]
            sba[idx] = complement_mapping_arr[back_byte]
            sba[rc_idx] = complement_mapping_arr[front_byte]
        reverse_complement_arr = sba
    else:
        reverse_complement_arr = np.zeros(len(sba), dtype=np.uint8)
        for idx in range(len(sba)):
            rc_idx = rc_idx = len(sba) - 1 - idx
            reverse_complement_arr[rc_idx] = complement_mapping_arr[sba[idx]]
    return reverse_complement_arr


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
        self._complement_mapping_arr = SequenceCollection._get_complement_mapping_array()
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

    @staticmethod
    def _get_complement_mapping_array():
        """
        Initialize the reverse_complement byte mapping array
        """
        # https://www.bioinformatics.org/sms/iupac
        # '$' is a special character that is used to mark the boundary between records in the
        # sequence byte array
        complement_mapping_dict = {
            "A": "T",
            "C": "G",
            "G": "C",
            "T": "A",
            "R": "Y",
            "Y": "R",
            "S": "S",
            "W": "W",
            "K": "M",
            "M": "K",
            "B": "V",
            "D": "H",
            "H": "D",
            "V": "B",
            "N": "N",
            "$": "$",
        }

        # build array mapping
        complement_mapping_arr = np.zeros(256, dtype=np.uint8)
        for key, val in complement_mapping_dict.items():
            complement_mapping_arr[ord(key)] = ord(val)
        return complement_mapping_arr

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
        total_seq_len = 0
        for record_name, seq in sequence_list:
            if len(seq) == 0:
                raise ValueError(
                    f"Each sequence in the collection must have length > 0.  Record '{record_name}' has a sequence lengt of 0"
                )
            total_seq_len += len(seq)
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

    def reverse_complement(self) -> np.array:
        """
        Reverse complement the sequence byte array.  Only valid if a single strand is loaded.
        """
        if self._strands_loaded == "both":
            raise ValueError(f"self._strands_loaded ({self._strands_loaded}) cannot be 'both'")

        if self._strands_loaded == "forward":
            self.revcomp_sba = self.forward_sba
            self.forward_sba = None
            reverse_complement(self.revcomp_sba, self._complement_mapping_arr, inplace=True)
            self._revcomp_sba_seq_starts = self._forward_sba_seq_starts
            self._forward_sba_seq_starts = None
            self._revcomp_sba_seq_starts = self._get_opposite_strand_sba_indices(
                self._revcomp_sba_seq_starts,
                len(self.revcomp_sba),
            )
            self._strands_loaded = "reverse_complement"
        elif self._strands_loaded == "reverse_complement":
            self.forward_sba = self.revcomp_sba
            self.revcomp_sba = None
            reverse_complement(self.forward_sba, self._complement_mapping_arr, inplace=True)
            self._forward_sba_seq_starts = self._revcomp_sba_seq_starts
            self._revcomp_sba_seq_starts = None
            self._forward_sba_seq_starts = self._get_opposite_strand_sba_indices(
                self._forward_sba_seq_starts,
                len(self.revcomp_sba),
            )
            self._strands_loaded = "forward"

        return

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
