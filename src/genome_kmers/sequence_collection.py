# from bisect import bisect_right
import pickle
import shelve
from collections import Counter
from pathlib import Path
from typing import Callable, List, Union

import h5py
import numpy as np
from numba import jit
from numba.core import types
from numba.typed import Dict


@jit
def bisect_right(a, x):
    """
    NOTE: this is a minor modification to the copy and paste from the python standard library. It
    is not possible to use the python standard library version with the @jit decorator, which is
    required since functions that call this function need to use the @jit decorator.

    Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@jit
def reverse_complement_sba(
    sba: np.array, complement_mapping_arr: np.array, inplace=False
) -> np.array:
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


@jit
def get_segment_num_from_sba_index(sba_idx: int, sba_strand: str, sba_seg_starts: np.array) -> int:
    """
    Get the segment number from the sequence byte array index.

    Args:
        sba_idx (int): sequence byte array index
        sba_strand (str): "forward" or "reverse_complement"
        sba_seg_starts (np.array): sba indices for each segment start (dtype=np.unit32)

    Returns:
        int: segment_num
    """
    # NOTE: if this is too slow in profiling, an alternative implementation is to generate a
    # look-up table that takes O(1) average look-up time if the distribution of sequence lengths
    # isn't too wide. Define an array of length (sba_len / N) and populate each index
    # (sba_idx // N) with the sba_idx.  Choose N to be small enough to ensure O(1) lookup time.
    # Will have bad memory usage in worst case (e.g. 1 length 1e7, 1e7 of length 1)

    # use a @jit decorated copy of Python's bisect function to do O(log(N)) search for the correct
    # record number using
    return bisect_right(sba_seg_starts, sba_idx) - 1


@jit
def get_forward_seq_idx(
    sba_idx: int,
    sba_strand: str,
    seg_sba_start_idx: int,
    seg_sba_end_idx: int,
    one_based: bool = False,
) -> int:
    """
    Get the forward sequence index given a sequence byte array index and segment start and and
    indices. Optionally returns a one-based sequence index.

    Args:
        sba_idx (int): sequence byte array index
        sba_strand (str, optional): sequence byte array strand.  If strands_loaded  is
            "both", then it must be specified as forward" or "reverse_complement".  Otherwise it
            is inferred. Defaults to None.
        seg_sba_start_idx (int): sequence byte array index for the start of the segment
        seg_sba_end_idx (int): sequence byte array index for the of the segment
        one_based (bool, optional): whether seq index return be one-based. Defaults to False.

    Raises:
        ValueError: if sba_idx, seg_sba_start_idx, or seg_sba_end_idx is not valid
        ValueError: if sba_strand is not recognized

    Returns:
        int: sequence forward strand index
    """
    # check that sba_idx, seg_sba_start_idx, and seg_sba_end_idx are valid
    if sba_idx < seg_sba_start_idx:
        raise ValueError(f"sba_idx ({sba_idx}) must be >= seg_sba_start_idx ({seg_sba_start_idx})")
    if sba_idx > seg_sba_end_idx:
        raise ValueError(f"sba_idx ({sba_idx}) must be <= seg_end_start_idx ({seg_sba_end_idx})")
    if seg_sba_start_idx > seg_sba_end_idx:
        raise ValueError(
            f"seg_sba_start_idx ({seg_sba_start_idx}) must be <= seg_sba_end_idx ({seg_sba_end_idx})"
        )
    if seg_sba_start_idx < 0:
        raise ValueError(f"seg_sba_start_idx ({seg_sba_start_idx}) must be > 0")

    # calculate seq_idx
    if sba_strand == "forward":
        seq_idx = sba_idx - seg_sba_start_idx
    elif sba_strand == "reverse_complement":
        seq_idx = seg_sba_end_idx - sba_idx
    else:
        raise ValueError(f"sba_strand ({sba_strand}) not recognized")

    # add one if one_based
    if one_based:
        seq_idx += 1

    return seq_idx


@jit
def get_sba_start_end_indices_for_segment(
    segment_num: int, sba_strand: str, sba_seg_starts: np.array, len_sba: int
) -> tuple[int, int]:
    """
    Given a segment number (and optionally sba_strand), return the first sba index and last sba
    index of the segment.

    Args:
        segment_num (int): segment number
        sba_strand (str, optional): sequence byte array strand ("forward", "reverse_complement",
            or "both"). Defaults to None.

    Raises:
        ValueError: segment_num out of bounds

    Returns:
        tuple[int, int]: sba_start_index, sba_end_index
    """
    # verify segment_num is valid
    if segment_num < 0:
        raise ValueError(f"segment_num ({segment_num}) is out of bounds")
    elif segment_num >= len(sba_seg_starts):
        raise ValueError(f"segment_num ({segment_num}) is out of bounds")

    # get start_index
    sba_start_index = sba_seg_starts[segment_num]
    if segment_num == len(sba_seg_starts) - 1:
        sba_end_index = len_sba - 1
    else:
        sba_end_index = sba_seg_starts[segment_num + 1] - 2

    return sba_start_index, sba_end_index


class SequenceCollection:
    """
    Holds all the information contained within a fasta file in a format conducive to
    kmer sorting.

    Terminology
    -----------
    .. code-block::


        sba: sequence byte array
        revcomp: reverse complement
        record: each header and its corresponding sequence is called a record.  record_num is based on
            the order that records are read in.  record_num does not change when reverse
            complemented
        segment: is the same as a record except that segment_num always starts leftmost.  i.e. the
            sba end index for segment N is always less than the sba end index for segment M > N
        forward_sba_idx: index in forward sequence byte array
        revcomp_sba_idx: index in reverse complement sequence byte array
        forward_seq_idx: 0-based index for a sequence on the forward strand
        revcomp_seq_idx: 0-based index for a sequence on the reverse complement strand

        Forward strand
        --------------
        record_num                   0           1                  2
        segment_num                  0           1                  2
        forward_record_name          A           B                  C
        forward_sba              [-------]$[------------]$[--------------------]
                                 |       | |            | |                    |
        forward_sba_seg_starts   0       | 10           | 25                   |
        forward_sba_seg_ends             8              23                     46

        Reverse complement strand
        -------------------------
        revcomp_sba_seg_ends                          21             36        46
        revcomp_sba_seg_starts   0                    | 23           | 38      |
                                 |                    | |            | |       |
        revcomp_sba              [====================]$[============]$[=======]
        revcomp_record_name                C                   B           A
        segment_num                        0                   1           2
        record_num                         2                   1           0

    Notes
    -----
    .. code-block::

        - a "$" is placed between each sequence so that you can determine if you've reached the end of
            a sequence without referencing the seg_starts / seg_ends array.
        - the collection must contain at least one sequence
        - all sequences in the collection must have a length > 0
        - duplicate record_names are not allowed

    Members
    -------
    .. code-block::

        forward_sba (np.array): forward sba (dtype=uint8)
        _forward_sba_seg_starts (np.array): value at index i gives the sba start index for the ith
            segment on the forward strand. (dtype=uint32)
        revcomp_sba (np.array): reverse complement sba (dtype=uint8)
        _revcomp_sba_seg_starts (np.array): value at index i gives the sba start index for the ith
            segment on the reverse complement strand. (dtype=uint32)
    """

    def __init__(
        self,
        fasta_file_path: Union[Path, None] = None,
        sequence_list: Union[list[tuple[str, str]], None] = None,
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
        # set default parameters based on user arguments
        self.forward_sba = None
        self._forward_sba_seg_starts = None
        self.forward_record_names = None
        self.revcomp_sba = None
        self._revcomp_sba_seg_starts = None
        self.revcomp_record_names = None
        self._strands_loaded = None
        self._fasta_file_path = None

        self._initialize_mapping_arrays()

        # if fasta_file_path and sequence_list are both None, do not do any additional initialization
        if fasta_file_path is None and sequence_list is None:
            return

        # check provided arguments
        if fasta_file_path is not None and sequence_list is not None:
            raise ValueError("Only one of fasta_file_path and sequence_list can be specified")
        if strands_to_load not in ("forward", "reverse_complement", "both"):
            raise ValueError(f"strands_to_load unrecognized ({strands_to_load})")

        # load sequence
        if fasta_file_path is not None:
            self._fasta_file_path = fasta_file_path
            self._initialize_from_fasta(fasta_file_path, strands_to_load)
        else:
            self._initialize_from_sequence_list(sequence_list, strands_to_load)

        return

    def __len__(self) -> int:
        """
        Count of sequence records in the collection
        """
        if self._strands_loaded == "forward" or self._strands_loaded == "both":
            return len(self._forward_sba_seg_starts)
        elif self._strands_loaded == "reverse_complement":
            return len(self._revcomp_sba_seg_starts)
        else:
            raise AssertionError(f"strands_loaded ({self._strands_loaded}) not recognized")

    def __str__(self) -> str:
        """
        Return fasta-formatted sequence.  If _strands_loaded == "forward" or "both", return the
        forward sequence.  Otherwise, return the reverse complement, but keep the record order
        the same as for the forward sequence.
        """
        sba_strand = (
            "reverse_complement" if self._strands_loaded == "reverse_complement" else "forward"
        )

        if sba_strand == "forward":
            sba = self.forward_sba
        elif sba_strand == "reverse_complement":
            sba = self.revcomp_sba

        lines = []
        for record_name, sba_seg_start_idx, sba_seg_end_idx in self.iter_records(sba_strand):
            seq = bytearray(sba[sba_seg_start_idx : sba_seg_end_idx + 1]).decode()
            lines.append(f">{record_name}")
            lines.append(seq)
        return "\n".join(lines)

    def sequence_length(self, record_num=None, record_name=None):
        """
        Return:
            If record_num is specified, then the length of record_num.
            If record_name is specified, then the length of record_num corresponding to
            record_name
            Otherwise, the total length of all sequences
        """
        if record_name is not None and record_num is not None:
            raise ValueError(
                f"record_num ({record_num}) and record_name ({record_name}) cannot both be specified"
            )

        # TODO: implement this function after carefully thinking through the functionality desired.
        #       record_num vs segment_num needs to be considered. This also likely needs to have
        #       strand specified (if using segment_num)
        raise NotImplementedError()

    def iter_records(self, sba_strand: str = None):
        """
        Yield (record_name, record_sba_start_idx, record_sba_end_idx) for all sequence records
        in order of increasing **record_num**.  i.e. if records are ordered as "chr1", "chr2",
        "chr3", they will be yielded in that same order regardless of whether the strand is
        "forward" or "reverse_complement".

        Args:
            sba_strand (str, optional): for which strand to yield records.  If
                self._strands_loaded == "both", it must be specified.  Otherwise it is determined
                from the loaded strand ("forward" or "reverse_complement").  Defaults to None.

        Yield:
            (record_name, record_sba_start_idx, record_sba_end_idx)
        """
        # decide which strand to use based on user parameter
        sba_strand = self._get_sba_strand_to_use(sba_strand)

        if sba_strand == "forward":
            for segment_num in range(len(self)):
                record_name = self.forward_record_names[segment_num]
                sba_seg_start_idx, sba_seg_end_idx = get_sba_start_end_indices_for_segment(
                    segment_num, sba_strand, self._forward_sba_seg_starts, len(self.forward_sba)
                )
                yield (record_name, sba_seg_start_idx, sba_seg_end_idx)
        elif sba_strand == "reverse_complement":
            # iterate in opposite order of segments to maintain correct record_num ordering
            for segment_num in range(len(self) - 1, -1, -1):
                record_name = self.revcomp_record_names[segment_num]
                sba_seg_start_idx, sba_seg_end_idx = get_sba_start_end_indices_for_segment(
                    segment_num, sba_strand, self._revcomp_sba_seg_starts, len(self.revcomp_sba)
                )
                yield (record_name, sba_seg_start_idx, sba_seg_end_idx)
        else:
            raise ValueError(f"sba_strand ({sba_strand}) must be 'forward' or 'reverse_complement'")
        return

    def strands_loaded(self) -> str:
        """
        Returns which strands are loaded.

        Returns:
            str: which strands are loaded "forward", "reverse_complement", "both"
        """
        return self._strands_loaded

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

    @staticmethod
    def _get_fasta_stats(fasta_file_path: Path) -> tuple[int, int]:
        """
        Read fasta file to determine number of records and total sequence length.

        Args:
            fasta_file_path (Path): fasta file

        Returns:
            tuple[int, int]: num_records, total_seq_len
        """
        num_records = 0
        total_seq_len = 0
        with open(fasta_file_path, "r") as input_file:
            for line in input_file:
                if line.startswith(">"):
                    num_records += 1
                else:
                    total_seq_len += len(line.strip())
        return num_records, total_seq_len

    @staticmethod
    def _get_fasta_record_name(line: str) -> str:
        """
        Get the fasta record name from a line starting with ">".  Use the same method as Bowtie,
        which reads all characters following ">" until the end of the line or whitespace.

        Args:
            line (str): fasta file line

        Raises:
            ValueError: if line does not start with ">"

        Returns:
            str: record_name
        """
        if not line.startswith(">"):
            raise ValueError("line does not start with '>'")
        record_name = line[1:].strip().split()[0]
        return record_name

    def _load_forward_sba_from_fasta(
        self, fasta_file_path: Path, num_records: int, total_seq_len: int
    ) -> tuple[np.array, np.array]:
        """
        Load forward sequence byte array from fasta file.

        Args:
            fasta_file_path (Path): fasta file to read
            num_records (int): number of records in the file
            total_seq_len (int): total length of all sequences contained in the fasta file

        Returns:
            tuple[np.array, np.array]: _description_
        """
        sba_len = total_seq_len + num_records - 1
        sba_seg_starts = np.zeros(num_records, dtype=np.uint32)
        sba = np.zeros(sba_len, dtype=np.uint8)
        record_names = []
        first_empty_idx = 0
        record_num = -1
        with open(fasta_file_path, "r") as input_file:
            for line in input_file:
                if line.startswith(">"):
                    record_num += 1

                    # if there is previous sequence, need to add a "$" to sba
                    if first_empty_idx != 0:
                        sba[first_empty_idx] = ord("$")
                        first_empty_idx += 1

                    # set the segment start index
                    sba_seg_starts[record_num] = first_empty_idx

                    # collect the record_name
                    record_name = SequenceCollection._get_fasta_record_name(line)
                    record_names.append(record_name)
                else:
                    sba_chunk = bytearray(line.strip().upper(), "utf-8")
                    sba[first_empty_idx : first_empty_idx + len(sba_chunk)] = sba_chunk
                    first_empty_idx += len(sba_chunk)

        if first_empty_idx != sba_len:
            raise AssertionError("After parsing the fasta file, we expect sba to be full")

        # check for any sequences with length zero
        empty_seq_found = (np.diff(sba_seg_starts) < 2).any()
        if empty_seq_found:
            raise ValueError(
                f"At least one empty sequence was found in the input file ({fasta_file_path})"
            )

        SequenceCollection._verify_record_names_are_unique(record_names)

        # verify that there are no unrecognized values in the sba
        values_in_sba = set(np.unique(sba))
        values_not_allowed = values_in_sba - self._allowed_uint8
        if values_not_allowed != set():
            raise ValueError(f"Sequence contains non-allowed characters! ({values_not_allowed})")

        return sba, sba_seg_starts, record_names

    def _initialize_from_fasta(self, fasta_file_path: Path, strands_to_load: str) -> None:
        """
        Initialize from SequenceCollection object from a fasta file.

        Args:
            fasta_file_path (Path): fasta file
            strands_to_load (str): which strand(s) to load into sequence byte arrays ("forward",
            "reverse_complement", or "both")

        Raises:
            ValueError: if strand is not recognized
        """
        if strands_to_load not in ("forward", "reverse_complement", "both"):
            raise ValueError(f"strands_to_load not recognized ({strands_to_load})")

        self.forward_sba = None
        self._forward_sba_seg_starts = None
        self.revcomp_sba = None
        self._revcomp_sba_seg_starts = None
        self.forward_record_names = None
        self.revcomp_record_names = None

        # determine the full size of the fasta file
        num_records, total_seq_len = self._get_fasta_stats(fasta_file_path)

        if strands_to_load == "forward" or strands_to_load == "both":
            self.forward_sba, self._forward_sba_seg_starts, self.forward_record_names = (
                self._load_forward_sba_from_fasta(fasta_file_path, num_records, total_seq_len)
            )

        if strands_to_load == "both":
            # take advantage of having forward_sba and _forward_sba_seg_starts already loaded
            # into memory.  We need only copy the array and reverse_complement.
            self.revcomp_sba = np.copy(self.forward_sba)
            reverse_complement_sba(self.revcomp_sba, self._complement_mapping_arr, inplace=True)

            self._revcomp_sba_seg_starts = self._get_opposite_strand_sba_start_indices(
                self._forward_sba_seg_starts,
                len(self.revcomp_sba),
            )

            self.revcomp_record_names = self.forward_record_names.copy()
            self.revcomp_record_names.reverse()

        elif strands_to_load == "reverse_complement":
            # load forward strand information and then reverse complement in place
            self.forward_sba, self._forward_sba_seg_starts, self.forward_record_names = (
                self._load_forward_sba_from_fasta(fasta_file_path, num_records, total_seq_len)
            )
            self._strands_loaded = "forward"
            self.reverse_complement()

        self._strands_loaded = strands_to_load

        return

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
    def _verify_record_names_are_unique(record_names):
        # verify that there are no repeated record names that would lead to ambiguity
        record_names_counter = Counter(record_names)
        if len(record_names) != len(record_names_counter):
            num_record_names_repeated = len(
                [1 for count in record_names_counter.values() if count > 1]
            )
            raise ValueError(
                f"sequence_list contains {num_record_names_repeated} repeated record_names"
            )
        return

    @staticmethod
    def _get_record_names_from_sequence_list(sequence_list: list[tuple[str, str]]) -> List[str]:
        """
        Get the list of record names from sequence_list.  Verify that there are no duplicates.

        Args:
            sequence_list (list[tuple[str, str]]): List of (seq_id, seq)
                tuples defining a sequence collection. seq_id is the sequences id (i.e.
                the header in a fasta file, such as "chr1"). seq is the string sequence
                (e.g. "ACTG"). Defaults to None. Must be specified if fasta_file_path
                is not.

        Returns:
        """
        record_names = [record_name for record_name, _ in sequence_list]
        SequenceCollection._verify_record_names_are_unique(record_names)
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
        self._forward_sba_seg_starts = None
        self.revcomp_sba = None
        self._revcomp_sba_seg_starts = None
        self.forward_record_names = None
        self.revcomp_record_names = None

        if strands_to_load == "forward" or strands_to_load == "both":
            self.forward_sba = self._get_sba_from_sequence_list(sequence_list)
            self._forward_sba_seg_starts = self._get_sba_starts_from_sequence_list(sequence_list)
            self.forward_record_names = self._get_record_names_from_sequence_list(sequence_list)

        if strands_to_load == "both":
            # take advantage of having forward_sba and _forward_sba_seg_starts already loaded
            # into memory.  We need only copy the array and reverse_complement.
            self.revcomp_sba = np.copy(self.forward_sba)
            reverse_complement_sba(self.revcomp_sba, self._complement_mapping_arr, inplace=True)

            self._revcomp_sba_seg_starts = self._get_opposite_strand_sba_start_indices(
                self._forward_sba_seg_starts,
                len(self.revcomp_sba),
            )

            self.revcomp_record_names = self.forward_record_names.copy()
            self.revcomp_record_names.reverse()

        elif strands_to_load == "reverse_complement":
            # load forward strand information and then reverse complement in place
            self.revcomp_sba = self._get_sba_from_sequence_list(sequence_list)
            reverse_complement_sba(self.revcomp_sba, self._complement_mapping_arr, inplace=True)

            self._revcomp_sba_seg_starts = self._get_sba_starts_from_sequence_list(sequence_list)
            self._revcomp_sba_seg_starts = self._get_opposite_strand_sba_start_indices(
                self._revcomp_sba_seg_starts,
                len(self.revcomp_sba),
            )

            self.revcomp_record_names = self._get_record_names_from_sequence_list(sequence_list)
            self.revcomp_record_names.reverse()

        self._strands_loaded = strands_to_load

        return

    def reverse_complement(self) -> np.array:
        """
        Reverse complement the sequence byte array.  Only valid if a single strand is loaded.
        """
        if self._strands_loaded == "both":
            raise ValueError(f"self._strands_loaded ({self._strands_loaded}) cannot be 'both'")

        if self._strands_loaded == "forward":
            # sba
            self.revcomp_sba = self.forward_sba
            self.forward_sba = None
            reverse_complement_sba(self.revcomp_sba, self._complement_mapping_arr, inplace=True)

            # sba segment starts
            self._revcomp_sba_seg_starts = self._forward_sba_seg_starts
            self._forward_sba_seg_starts = None
            self._revcomp_sba_seg_starts = self._get_opposite_strand_sba_start_indices(
                self._revcomp_sba_seg_starts,
                len(self.revcomp_sba),
            )

            # record names
            self.revcomp_record_names = self.forward_record_names
            self.revcomp_record_names.reverse()
            self.forward_record_names = None

            self._strands_loaded = "reverse_complement"

        elif self._strands_loaded == "reverse_complement":
            # sba
            self.forward_sba = self.revcomp_sba
            self.revcomp_sba = None
            reverse_complement_sba(self.forward_sba, self._complement_mapping_arr, inplace=True)

            # sba segment starts
            self._forward_sba_seg_starts = self._revcomp_sba_seg_starts
            self._revcomp_sba_seg_starts = None
            self._forward_sba_seg_starts = self._get_opposite_strand_sba_start_indices(
                self._forward_sba_seg_starts,
                len(self.forward_sba),
            )

            # record names
            self.forward_record_names = self.revcomp_record_names
            self.forward_record_names.reverse()
            self.revcomp_record_names = None

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

    def get_record_loc_from_sba_index(
        self, sba_idx: int, sba_strand: str = None, one_based: bool = False
    ) -> tuple[str, str, int]:
        """
        Get the sequence location (strand, record_name, seq_idx) from the sequence byte array index

        Args:
            sba_idx (int): sequence byte array index
            sba_strand (str, optional): sequence byte array strand.  If strands_loaded  is
                "both", then it must be specified as forward" or "reverse_complement".  Otherwise it
                is inferred. Defaults to None.
            one_based (bool, optional): whether seq index return be one-based. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            tuple[str, str, int]: _description_
        """
        # decide which strand to use based on user parameter
        sba_strand = self._get_sba_strand_to_use(sba_strand)

        # get all loc info
        if sba_strand == "forward":
            segment_num = get_segment_num_from_sba_index(
                sba_idx, sba_strand, self._forward_sba_seg_starts
            )
            record_name = self.forward_record_names[segment_num]
            seg_sba_start_idx, seg_sba_end_idx = get_sba_start_end_indices_for_segment(
                segment_num, sba_strand, self._forward_sba_seg_starts, len(self.forward_sba)
            )
        elif sba_strand == "reverse_complement":
            segment_num = get_segment_num_from_sba_index(
                sba_idx, sba_strand, self._revcomp_sba_seg_starts
            )
            record_name = self.revcomp_record_names[segment_num]
            seg_sba_start_idx, seg_sba_end_idx = get_sba_start_end_indices_for_segment(
                segment_num, sba_strand, self._revcomp_sba_seg_starts, len(self.revcomp_sba)
            )
        else:
            raise ValueError(f"sba_strand ({sba_strand}) not recognized")

        seq_idx = get_forward_seq_idx(
            sba_idx, sba_strand, seg_sba_start_idx, seg_sba_end_idx, one_based=one_based
        )

        strand = "+" if sba_strand == "forward" else "-"

        return (strand, record_name, seq_idx)

    def get_record_name_from_sba_index(self, sba_idx: int, sba_strand: str = None) -> str:
        """
        Get the sequence record number from the sequence byte array index.

        Args:
            sba_idx (int): sequence byte array index
            sba_strand (str, optional): for which strand is the sba_idx defined ("forward" or
                "reverse_complement").  Must be defined when SequenceCollection has both
                strands loaded.  If specified when only a single strand has been loaded, it will
                verify that it matches what is expected.  If set to None, it will automatically
                detect the strand that was loaded.  Defaults to None.

        Returns:
            record_name (str):
        """
        # decide which strand to use based on user parameter
        sba_strand = self._get_sba_strand_to_use(sba_strand)

        if sba_strand == "forward":
            segment_num = get_segment_num_from_sba_index(
                sba_idx, sba_strand, self._forward_sba_seg_starts
            )
            record_name = self.forward_record_names[segment_num]
        elif sba_strand == "reverse_complement":
            segment_num = get_segment_num_from_sba_index(
                sba_idx, sba_strand, self._revcomp_sba_seg_starts
            )
            record_name = self.revcomp_record_names[segment_num]
        else:
            raise ValueError(f"sba_strand ({sba_strand}) not recognized")

        return record_name

    def _get_sba_strand_to_use(self, sba_strand: str) -> str:
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

        sba_strand_to_use = self._strands_loaded if self._strands_loaded != "both" else sba_strand
        return sba_strand_to_use

    def get_segment_num_from_sba_index(self, sba_idx: int, sba_strand: str = None) -> int:
        """
        Get the segment number from the sequence byte array index defined on sba_strand
        (attempt to automatically detect the strand if not specified)

        Args:
            sba_idx (int): sequence byte array index
            sba_strand (str, optional): for which strand is the sba_idx defined ("forward" or
                "reverse_complement").  Must be defined when SequenceCollection has both
                strands loaded.  If specified when only a single strand has been loaded, it will
                verify that it matches what is expected.  If set to None, it will automatically
                detect the strand that was loaded.  Defaults to None.

        Returns:
            segment_num (int):
        """
        # decide which strand to use based on user parameter
        sba_strand = self._get_sba_strand_to_use(sba_strand)

        # get the segment number
        if sba_strand == "forward":
            if sba_idx < 0 or sba_idx >= len(self.forward_sba):
                raise IndexError(f"sba_idx ({sba_idx}) is out of bounds")
            segment_num = get_segment_num_from_sba_index(
                sba_idx, sba_strand, self._forward_sba_seg_starts
            )
        elif sba_strand == "reverse_complement":
            if sba_idx < 0 or sba_idx >= len(self.revcomp_sba):
                raise IndexError(f"sba_idx ({sba_idx}) is out of bounds")
            segment_num = get_segment_num_from_sba_index(
                sba_idx, sba_strand, self._revcomp_sba_seg_starts
            )

        return segment_num

    def get_sba_start_end_indices_for_segment(
        self, segment_num: int, sba_strand: str = None
    ) -> tuple[int, int]:
        """
        Given a segment number (and optionally sba_strand), return the first sba index and
        last sba index of the segment.

        Args:
            segment_num (int): segment number
            sba_strand (str, optional): sequence byte array strand ("forward",
                "reverse_complement", or "both"). Defaults to None.

        Raises:
            ValueError: segment_num out of bounds

        Returns:
            tuple[int, int]: sba_start_index, sba_end_index
        """
        # set sba_strand to either "forward" or "reverse_complement" based provided argument
        sba_strand = self._get_sba_strand_to_use(sba_strand)

        if sba_strand == "forward":
            sba_seg_starts = self._forward_sba_seg_starts
            sba = self.forward_sba
        elif sba_strand == "reverse_complement":
            sba_seg_starts = self._revcomp_sba_seg_starts
            sba = self.revcomp_sba

        # verify segment_num is allowed
        if segment_num < 0:
            raise ValueError(f"segment_num ({segment_num}) is out of bounds")
        elif segment_num >= len(sba_seg_starts):
            raise ValueError(f"segment_num ({segment_num}) is out of bounds")

        # get start_index
        sba_start_index = sba_seg_starts[segment_num]
        if segment_num == len(sba_seg_starts) - 1:
            sba_end_index = len(sba) - 1
        else:
            sba_end_index = sba_seg_starts[segment_num + 1] - 2

        return sba_start_index, sba_end_index

    def generate_get_record_info_from_sba_index_func(self, one_based: bool = False) -> Callable:
        """
        Generate a function that returns record info when provided a sequence byte array index

        Args:
            one_based (bool, optional): Should sequence indices by one based. Defaults to False.

        Raises:
            ValueError: sba_strand not recognized

        Returns:
            Callable: get_record_info_from_sba_index
        """
        # NOTE: below, record_names is caste into a tuple to avoid the following numba error
        #   Exception has occurred: NumbaNotImplementedError       (note: full exception trace is shown but execution is paused at: <module>)
        #   Failed in nopython mode pipeline (step: native lowering)
        #   <numba.core.base.OverloadSelector object at 0x7f4e9cf54880>, (List(unicode_type, True),)

        sba_strand = self._get_sba_strand_to_use(self.strands_loaded())

        # set record_names, sba_seg_starts, seq_strand
        if sba_strand == "forward":
            record_names = tuple(self.forward_record_names)
            sba_seg_starts = self._forward_sba_seg_starts
            seq_strand = "+"
            len_sba = len(self.forward_sba)
        elif sba_strand == "reverse_complement":
            record_names = tuple(self.revcomp_record_names)
            sba_seg_starts = self._revcomp_sba_seg_starts
            seq_strand = "-"
            len_sba = len(self.revcomp_sba)
        else:
            raise ValueError(f"sba_strand ({sba_strand}) not recognized")

        @jit
        def get_record_info_from_sba_index(sba_idx: int) -> tuple[int, int, str, str, int]:
            """
            Given a sequence byte array index, return record info.

            Args:
                sba_idx (int): sequence byte array index

            Raises:
                ValueError: _description_

            Returns:
                tuple[int, int, str, str, int]:
                    seg_sba_start_idx, seg_sba_end_idx, seq_strand, seq_record_name, seq_start_idx
            """
            # get the segment number that contains sba_idx
            seg_num = get_segment_num_from_sba_index(sba_idx, sba_strand, sba_seg_starts)

            # get the segment start and end indices in the sequence byte array
            seg_sba_start_idx, seg_sba_end_idx = get_sba_start_end_indices_for_segment(
                seg_num, sba_strand, sba_seg_starts, len_sba
            )

            # get the sequence index (on the forward strand)
            seq_start_idx = get_forward_seq_idx(
                sba_idx, sba_strand, seg_sba_start_idx, seg_sba_end_idx, one_based=one_based
            )

            # get record name
            seq_record_name = record_names[seg_num]

            return (
                seg_num,
                seg_sba_start_idx,
                seg_sba_end_idx,
                seq_strand,
                seq_record_name,
                seq_start_idx,
            )

        return get_record_info_from_sba_index

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        # NOTE: _fasta_file_path is not required to match for SequenceCollection objects to be equal
        # forward_sba
        if self.forward_sba is None and other.forward_sba is not None:
            return False
        elif self.forward_sba is not None and other.forward_sba is None:
            return False
        elif not np.array_equal(self.forward_sba, other.forward_sba):
            return False

        # _forward_sba_seg_starts
        if self._forward_sba_seg_starts is None and other._forward_sba_seg_starts is not None:
            return False
        elif self._forward_sba_seg_starts is not None and other._forward_sba_seg_starts is None:
            return False
        elif not np.array_equal(self._forward_sba_seg_starts, other._forward_sba_seg_starts):
            return False

        # forward_record_names
        if self.forward_record_names is None and other.forward_record_names is not None:
            return False
        elif self.forward_record_names is not None and other.forward_record_names is None:
            return False
        elif self.forward_record_names != other.forward_record_names:
            return False

        # revcomp_sba
        if self.revcomp_sba is None and other.revcomp_sba is not None:
            return False
        elif self.revcomp_sba is not None and other.revcomp_sba is None:
            return False
        elif not np.array_equal(self.revcomp_sba, other.revcomp_sba):
            return False

        # _revcomp_sba_seg_starts
        if self._revcomp_sba_seg_starts is None and other._revcomp_sba_seg_starts is not None:
            return False
        elif self._revcomp_sba_seg_starts is not None and other._revcomp_sba_seg_starts is None:
            return False
        elif not np.array_equal(self._revcomp_sba_seg_starts, other._revcomp_sba_seg_starts):
            return False

        # revcomp_record_names
        if self.revcomp_record_names is None and other.revcomp_record_names is not None:
            return False
        elif self.revcomp_record_names is not None and other.revcomp_record_names is None:
            return False
        elif self.revcomp_record_names != other.revcomp_record_names:
            return False

        # _strands_loaded
        if self._strands_loaded is None and other._strands_loaded is not None:
            return False
        elif self._strands_loaded is not None and other._strands_loaded is None:
            return False
        elif self._strands_loaded != other._strands_loaded:
            return False

        # if this is reached, then they are equal
        return True

    @staticmethod
    def _set_for_export(value, value_if_none):
        """
        Helper function that converts a None value into the appropriate indicator value if value
        equals value_if_none. This is used to help deal with the HDF5 format not handling null
        values.

        Args:
            value: value we want to pass to serialize with HDF5
            value_if_none: what to pass to HDF5 instead of value if value is None

        Returns:
            value_if_none if value is None else value
        """
        if value is None:
            return value_if_none
        else:
            return value

    @staticmethod
    def _correct_import(value, value_if_none):
        """
        Helper function that converts a value read from HDF5 to None if it equal to the indicator
        value value_if_none. This is used to help deal with the HDF5 format not handling null
        values.

        Args:
            value: value we want to pass to serialize with HDF5
            value_if_none: what to pass to HDF5 instead of value if value is None

        Returns:
            None if value == value_if_none else value
        """
        if isinstance(value, np.ndarray):
            if value.shape == (0,):
                return None
        elif value == value_if_none:
            return None
        return value

    def save(self, save_file_path: Path, mode: str = "a", format: str = "hdf5") -> None:
        """
        Save SequenceCollection object to file.

        Args:
            save_file_path (Path): path for saved file
            format (str, optional): "hdf5" or "shelve". Defaults to "hdf5".
            mode (str, optional): mode with which to open file and save information. "w" for write
                or "a" for append. Defaults to "a".

        Raises:
            ValueError: format not recognized
        """
        if format == "hdf5":
            self._save_hdf5(save_file_path, mode=mode)
        elif format == "shelve":
            self._save_shelve(save_file_path)
        else:
            raise ValueError(f"format ({format}) not recognized")

    def load(self, load_file_path: Path, format: str = "hdf5"):
        """
        Load SequenceCollection object from saved file.

        Args:
            load_file_path (Path): path to file to load.
            format (str, optional): "hdf5" or "shelve". Defaults to "hdf5".

        Raises:
            ValueError: format not recognized
        """
        if format == "hdf5":
            self._load_h5py(load_file_path)
        elif format == "shelve":
            self._load_shelve(load_file_path)
        else:
            raise ValueError(f"format ({format}) not recognized")

    def _save_hdf5(self, save_file_path: Path, mode: str = "a") -> None:
        """
        Save SequenceCollection object information to HDF5 file format.

        Args:
            save_file_path (Path): path for saved file
            mode (str, optional): mode with which to open file and save information. "w" for write
                or "a" for append. Defaults to "a".
        """
        with h5py.File(save_file_path, mode) as file:
            grp = file.create_group("seq_coll")

            # hdf5 does not accept None values.  Correct them before exporting.
            forward_sba = self._set_for_export(self.forward_sba, np.array([], dtype=np.uint8))
            _forward_sba_seg_starts = self._set_for_export(self._forward_sba_seg_starts, [])
            forward_record_names = self._set_for_export(self.forward_record_names, [])
            revcomp_sba = self._set_for_export(self.revcomp_sba, np.array([], dtype=np.uint8))
            _revcomp_sba_seg_starts = self._set_for_export(self._revcomp_sba_seg_starts, [])
            revcomp_record_names = self._set_for_export(self.revcomp_record_names, [])
            _strands_loaded = self._set_for_export(self._strands_loaded, "")
            _fasta_file_path = self._set_for_export(self._fasta_file_path, "")

            # save them to file
            grp["forward_sba"] = forward_sba
            grp["_forward_sba_seg_starts"] = _forward_sba_seg_starts
            grp["forward_record_names"] = forward_record_names
            grp["revcomp_sba"] = revcomp_sba
            grp["_revcomp_sba_seg_starts"] = _revcomp_sba_seg_starts
            grp["revcomp_record_names"] = revcomp_record_names
            grp["_strands_loaded"] = _strands_loaded
            grp["_fasta_file_path"] = str(_fasta_file_path)

            # do not save the members set by _initialize_mapping_arrays()

        return

    def _load_h5py(self, load_file_path: Path):
        """
        Load SequenceCollection object information from HDF5 file format.

        Args:
            load_file_path (Path): path to file to load.
        """
        with h5py.File(load_file_path, "r") as file:
            grp = file["seq_coll"]
            empty_sba = np.array([], dtype=np.uint8)

            # read values from file
            self.forward_sba = self._correct_import(grp["forward_sba"][:], empty_sba)
            self._forward_sba_seg_starts = self._correct_import(
                grp["_forward_sba_seg_starts"][:], []
            )
            self.forward_record_names = [val.decode("utf-8") for val in grp["forward_record_names"]]
            self.forward_record_names = self._correct_import(self.forward_record_names, [])

            self.revcomp_sba = self._correct_import(grp["revcomp_sba"][:], empty_sba)
            self._revcomp_sba_seg_starts = self._correct_import(
                grp["_revcomp_sba_seg_starts"][:], []
            )
            self.revcomp_record_names = [val.decode("utf-8") for val in grp["revcomp_record_names"]]
            self.revcomp_record_names = self._correct_import(self.revcomp_record_names, [])

            self._strands_loaded = self._correct_import(
                grp["_strands_loaded"][()].decode("utf-8"), ""
            )
            self._fasta_file_path = self._correct_import(
                grp["_fasta_file_path"][()].decode("utf-8"), ""
            )
            if self._fasta_file_path is not None:
                self._fasta_file_path = Path(self._fasta_file_path)

            # initialize mapping arrays
            self._initialize_mapping_arrays()

        return

    def _save_shelve(self, save_file_path: Path) -> None:
        """
        Save SequenceCollection object information to shelve file format.

        Args:
            save_file_path (Path): path for saved file
        """
        with shelve.open(save_file_path, protocol=pickle.DEFAULT_PROTOCOL) as db:
            db["seq_coll.forward_sba"] = self.forward_sba
            db["seq_coll._forward_sba_seg_starts"] = self._forward_sba_seg_starts
            db["seq_coll.forward_record_names"] = self.forward_record_names
            db["seq_coll.revcomp_sba"] = self.revcomp_sba
            db["seq_coll._revcomp_sba_seg_starts"] = self._revcomp_sba_seg_starts
            db["seq_coll.revcomp_record_names"] = self.revcomp_record_names
            db["seq_coll._strands_loaded"] = self._strands_loaded
            db["seq_coll._fasta_file_path"] = self._fasta_file_path

            # do not save the members set by _initialize_mapping_arrays()

        return

    def _load_shelve(self, load_file_path: Path):
        """
        Load SequenceCollection object information from shelve file format.

        Args:
            load_file_path (Path): path to file to load.
        """
        with shelve.open(load_file_path) as db:
            self.forward_sba = db["seq_coll.forward_sba"]
            self._forward_sba_seg_starts = db["seq_coll._forward_sba_seg_starts"]
            self.forward_record_names = db["seq_coll.forward_record_names"]
            self.revcomp_sba = db["seq_coll.revcomp_sba"]
            self._revcomp_sba_seg_starts = db["seq_coll._revcomp_sba_seg_starts"]
            self.revcomp_record_names = db["seq_coll.revcomp_record_names"]
            self._strands_loaded = db["seq_coll._strands_loaded"]
            self._fasta_file_path = db["seq_coll._fasta_file_path"]

            self._initialize_mapping_arrays()
        return
