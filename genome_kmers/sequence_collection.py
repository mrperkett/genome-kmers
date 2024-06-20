from bisect import bisect_right
from pathlib import Path
from typing import List

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

    @staticmethod
    def _get_required_sba_length_from_sequence_list(sequence_list: list[tuple[str, str]]) -> int:
        """
        Calculate the size of the sequence byte array to allocate.  Note that a '$' is placed
        between each sequence, which accounts for the length beyond just sequence.

        Args:
            sequence_list (list[tuple[str, str]]): List of (seq_id, seq)
                tuples defining a sequence collection. seq_id is the sequences id (i.e.
                the header in a fasta file, such as "chr1"). seq is the string sequence
                (e.g. "ACTG"). Defaults to None. Must be specified if fasta_file_path
                is not.

        Raises:
            ValueError: raised if there is a sequence in sequence_list with length 0

        Returns:
            sba_length (int): length required for sequence byte array
        """
        total_seq_len = 0
        for record_name, seq in sequence_list:
            if len(seq) == 0:
                raise ValueError(
                    f"Each sequence in the collection must have length > 0.  Record '{record_name}' has a sequence lengt of 0"
                )
            total_seq_len += len(seq)
        sba_length = total_seq_len + len(sequence_list) - 1
        return sba_length

    def _get_sba_from_sequence_list(self, sequence_list: list[tuple[str, str]]) -> np.array:
        """
        Generate a sequence byte array from a sequence list.

        Args:
            sequence_list (list[tuple[str, str]]): List of (seq_id, seq)
                tuples defining a sequence collection. seq_id is the sequences id (i.e.
                the header in a fasta file, such as "chr1"). seq is the string sequence
                (e.g. "ACTG"). Defaults to None. Must be specified if fasta_file_path
                is not.

        Raises:
            ValueError: raised when sequence contains non-allowed values

        Returns:
            sba (np.array): sequence byte array
        """
        sba_length = SequenceCollection._get_required_sba_length_from_sequence_list(sequence_list)
        sba = np.zeros(sba_length, dtype=np.uint8)
        last_filled_idx = -1
        for i, (_, seq) in enumerate(sequence_list):
            start_idx = last_filled_idx + 1
            sba[start_idx : start_idx + len(seq)] = bytearray(seq, "utf-8")
            last_filled_idx = start_idx + len(seq) - 1

            # place a '$' between loaded sequences
            if i != len(sequence_list) - 1:
                last_filled_idx += 1
                sba[last_filled_idx] = ord("$")

        # verify that there are no unrecognized values in the sba
        values_in_sba = set(np.unique(sba))
        values_not_allowed = values_in_sba - self._allowed_uint8
        if values_not_allowed != set():
            raise ValueError(f"Sequence contains non-allowed characters! ({values_not_allowed})")

        return sba

    @staticmethod
    def _get_sba_starts_from_sequence_list(sequence_list: list[tuple[str, str]]) -> np.array:
        """
        Generate an array of sequence start indices within the sequence byte array from
        sequence_list.

        Args:
            sequence_list (list[tuple[str, str]]): List of (seq_id, seq)
                tuples defining a sequence collection. seq_id is the sequences id (i.e.
                the header in a fasta file, such as "chr1"). seq is the string sequence
                (e.g. "ACTG"). Defaults to None. Must be specified if fasta_file_path
                is not.

        Returns:
            sba_starts (np.array): array with sba index for the start of the sequence (dtype=uint32)
        """
        sba_seq_starts = np.zeros(len(sequence_list), dtype=np.uint32)
        last_filled_idx = -1
        for i, (_, seq) in enumerate(sequence_list):
            start_idx = last_filled_idx + 1
            sba_seq_starts[i] = start_idx
            last_filled_idx = start_idx + len(seq) - 1
            # place a '$' between loaded sequences
            if i != len(sequence_list) - 1:
                last_filled_idx += 1
        return sba_seq_starts

    @staticmethod
    def _get_record_names_from_sequence_list(sequence_list: list[tuple[str, str]]) -> List[str]:
        """

        Args:
            sequence_list (list[tuple[str, str]]): List of (seq_id, seq)
                tuples defining a sequence collection. seq_id is the sequences id (i.e.
                the header in a fasta file, such as "chr1"). seq is the string sequence
                (e.g. "ACTG"). Defaults to None. Must be specified if fasta_file_path
                is not.

        Returns:
        """
        record_names = [record_name for record_name, _ in sequence_list]
        return record_names

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
        if strands_to_load not in ("forward", "reverse_complement", "both"):
            raise ValueError(f"strands_to_load not recognized ({strands_to_load})")

        self.forward_sba = None
        self._forward_sba_seq_starts = None
        self.revcomp_sba = None
        self._revcomp_sba_seq_starts = None

        if strands_to_load == "forward" or strands_to_load == "both":
            self.forward_sba = self._get_sba_from_sequence_list(sequence_list)
            self._forward_sba_seq_starts = self._get_sba_starts_from_sequence_list(sequence_list)

        if strands_to_load == "both":
            # take advantage of having forward_sba  and _forward_sba_seq_starts already loaded
            # into memory.  We need only copy the array and reverse_complement.
            self.revcomp_sba = np.copy(self.forward_sba)
            reverse_complement(self.revcomp_sba, self._complement_mapping_arr, inplace=True)

            self._revcomp_sba_seq_starts = self._get_opposite_strand_sba_start_indices(
                self._forward_sba_seq_starts,
                len(self.revcomp_sba),
            )

        elif strands_to_load == "reverse_complement":
            # load forward strand information and then reverse complement in place
            self.revcomp_sba = self._get_sba_from_sequence_list(sequence_list)
            reverse_complement(self.revcomp_sba, self._complement_mapping_arr, inplace=True)

            self._revcomp_sba_seq_starts = self._get_sba_starts_from_sequence_list(sequence_list)
            self._revcomp_sba_seq_starts = self._get_opposite_strand_sba_start_indices(
                self._revcomp_sba_seq_starts,
                len(self.revcomp_sba),
            )

        self.record_names = self._get_record_names_from_sequence_list(sequence_list)
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
            self._revcomp_sba_seq_starts = self._get_opposite_strand_sba_start_indices(
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
            self._forward_sba_seq_starts = self._get_opposite_strand_sba_start_indices(
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

    @staticmethod
    def _get_opposite_strand_sba_start_indices(sba_starts: np.array, sba_len: int) -> np.array:
        """
        Get the sba_start_indices for the opposite strand.  A sba_start_index is defined as the
        leftmost inclusive start index of the corresponding sequence.  The sba_start_indices are
        ordered from smallest to largest.

        Args:
            sba_starts (np.array): sequence byte array start indices dtype=np.unit32
            sba_len (int): total length of the sequence byte array

        Returns:
            opposite_strand_start_indices (np.array): sequence byte array start indices for
                the reverse complement.
        """
        # NOTE: The start of each sequence on the opposite strand is the current end of the
        # sequence.  Also, the order will need to be reversed to keep it in ascending order
        sba_end_indices = np.copy(sba_starts)
        if len(sba_end_indices) > 1:
            sba_end_indices[:-1] = sba_end_indices[1:] - 2
        sba_end_indices[-1] = sba_len - 1
        opposite_strand_start_indices = SequenceCollection._get_opposite_strand_sba_indices(
            np.flip(sba_end_indices), sba_len
        )
        return opposite_strand_start_indices

    def record_name_from_sba_index(self, ba_idx, ba_strand=None):
        """
        NOTE: this may need to be sped up using nb.jit.
        """
        pass

    @staticmethod
    def _get_record_num_from_sba_index(sba_seq_starts: np.array, sba_idx: int) -> int:
        """
        Get the sequence record number from the sequence byte array index.

        NOTE: no checking of argument values is done in this function.  If checking is required, it
        should be done through the wrapper function get_record_num_from_sba_idx.

        Args:
            sba_seq_starts (np.array): sequence byte array start indices dtype=np.unit32
            sba_idx (int): sequence byte array index

        Returns:
            record_num (int):
        """
        # TODO: if this is too slow in profiling, an alternative implementation is to generate a
        # look-up table that takes O(1) average look-up time if the distribution of sequence lengths
        # isn't too wide. Define an array of length (sba_len / N) and populate each index
        # (sba_idx // N) with the sba_idx.  Choose N to be small enough to ensure O(1) lookup time.
        # Will have bad memory usage in worst case (e.g. 1 length 1e7, 1e7 of length 1)

        # use Python's bisect function to do O(log(N)) search for the correct record number using
        # _forward_sba_seq_starts or _revcomp_sba_seq_starts
        return bisect_right(sba_seq_starts, sba_idx) - 1

    def get_record_num_from_sba_index(self, sba_idx: int, sba_strand: str = None) -> int:
        """
        Get the sequence record number from the sequence byte array index defined on sba_strand
        (attempt to automatically detect the strand if not specified)

        Args:
            sba_idx (int): sequence byte array index
            sba_strand (str, optional): for which strand is the sba_idx defined ("forward" or
                "reverse_complement").  Must be defined when SequenceCollection has both
                strands loaded.  If specified when only a single strand has been loaded, it will
                verify that it matches what is expected.  If set to None, it will automatically
                detect the strand that was loaded.  Defaults to None.

        Returns:
            record_num (int):
        """
        # sba_strand only needs to be specified for self._strands_loaded == "both".  If provided
        # for "forward" or "reverse_complement", it is verified to match
        if sba_strand is not None:
            if sba_strand == "forward":
                if self._strands_loaded == "reverse_complement":
                    raise ValueError(
                        f"sba_strand ({sba_strand}) does not match _strands_loaded ({self._strands_loaded})"
                    )
            elif sba_strand == "reverse_complement":
                if self._strands_loaded == "forward":
                    raise ValueError(
                        f"sba_strand ({sba_strand}) does not match _strands_loaded ({self._strands_loaded})"
                    )
            else:
                raise ValueError(f"sba_strand ({sba_strand}) not recognized")
        if self._strands_loaded == "both" and sba_strand is None:
            raise ValueError("sba_strand must be specified when both strands are loaded")

        # get the record number
        if self._strands_loaded == "forward" or sba_strand == "forward":
            if sba_idx < 0 or sba_idx >= len(self.forward_sba):
                raise IndexError(f"sba_idx ({sba_idx}) is out of bounds")
            record_num = SequenceCollection._get_record_num_from_sba_index(
                self._forward_sba_seq_starts, sba_idx
            )
        elif self._strands_loaded == "reverse_complement" or sba_strand == "reverse_complement":
            if sba_idx < 0 or sba_idx >= len(self.revcomp_sba):
                raise IndexError(f"sba_idx ({sba_idx}) is out of bounds")
            record_num = SequenceCollection._get_record_num_from_sba_index(
                self._revcomp_sba_seq_starts, sba_idx
            )

        return record_num
