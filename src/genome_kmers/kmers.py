import numpy as np

from genome_kmers.sequence_collection import SequenceCollection


class Kmers:
    """
    Defines memory-efficient kmers calculations on a genome.
    """

    def __init__(
        self,
        sequence_collection: SequenceCollection,
        min_kmer_len: tuple[int, None] = None,
        max_kmer_len: tuple[int, None] = None,
        source_strand: str = "forward",
        track_strands_separately: bool = False,
        method: str = "single_pass",
    ) -> None:
        """
        Initialize Kmers

        Args:
            sequence_collection (SequenceCollection): the sequence collection on which kmers are
                defined
            min_kmer_len (tuple[int, None], optional): kmers below this size are not considered.
                Defaults to None.
            max_kmer_len (tuple[int, None], optional): kmers above this size are not considered.
                Defaults to None.
            source_strand (str, optional): strand(s) on which kmers are defined ("forward",
                "reverse_complement", "both"). Defaults to "forward".
            track_strands_separately (bool, optional): if source_strand is set to "both", this
                specifies whether kmers should be tracked separately by strand (which is
                required for certain kmer filters)
            method (str, optional): which method to use for initialization.  "single_pass" runs
                faster, but temporarily uses more RAM (up to 2x).  "double_pass" runs slower (~2x)
                but uses less memory (as much as half).  Defaults to "single_pass".

        Raises:
            ValueError: invalid arguments
            NotImplementedError: yet to be implemented functionality
        """
        # verify that arguments are valid
        if source_strand not in ("forward", "reverse_complement", "both"):
            raise ValueError(f"source_strand ({source_strand}) not recognized")
        if track_strands_separately not in (True, False):
            raise ValueError(
                f"track_strands_separately must be True or False, but it has value {track_strands_separately}"
            )
        if source_strand != "both" and track_strands_separately:
            raise ValueError(
                f"track_strands_separately can only be true if source_strand is 'both', but it is '{source_strand}'"
            )
        if min_kmer_len is not None and min_kmer_len < 1:
            raise ValueError(f"min_kmer_len ({min_kmer_len}) must be greater than zero")
        if max_kmer_len is not None:
            if max_kmer_len < 1:
                raise ValueError(f"max_kmer_len ({max_kmer_len}) must be greater than zero")
            if min_kmer_len is not None and max_kmer_len < min_kmer_len:
                raise ValueError(
                    f"max_kmer_len ({max_kmer_len}) is less than min_kmer_len ({min_kmer_len})"
                )

        # count number of records and sequence lengths in sequence_collection
        seq_lengths = []
        min_seq_len = None
        num_records = 0
        record_generator = sequence_collection.iter_records()
        for _, sba_seg_start_idx, sba_seg_end_idx in record_generator:
            seq_length = sba_seg_end_idx - sba_seg_start_idx + 1
            seq_lengths.append(seq_length)
            if min_seq_len is None or seq_length < min_seq_len:
                min_seq_len = seq_length
            num_records += 1

        # verify that arguments are valid given the sequence_collection
        if num_records == 0:
            raise ValueError(f"sequence_collection is empty")
        if min_kmer_len is not None and min_kmer_len > min_seq_len:
            raise ValueError(
                f"min_kmer_len ({min_kmer_len}) must be <= the shortest sequence length ({min_seq_len})"
            )
        if sequence_collection.strands_loaded() != source_strand:
            # for now, require that source_strand matches what is in the SequenceCollection for ease
            # of implementation
            raise ValueError(
                f"source_strand ({source_strand}) does not match sequence_collection loaded strand ({sequence_collection.strands_loaded()})"
            )

        # check that argument values have implemented functionality
        if track_strands_separately:
            raise NotImplementedError(
                f"This function has not been implemented for track_strands_separately = '{track_strands_separately}'"
            )
        if source_strand != "forward":
            raise NotImplementedError(
                f"This function has not been implemented for source_strand = '{source_strand}'"
            )

        # set member variables based on arguments
        self.sequence_collection = sequence_collection
        if min_kmer_len is None:
            self.min_kmer_len = 1
        else:
            self.min_kmer_len = min_kmer_len
        self.max_kmer_len = max_kmer_len
        self.kmer_source_strand = source_strand
        self.track_strands_separately = track_strands_separately

        self._is_initialized = False
        self._is_set = False
        self._is_sorted = False
        self.kmer_sba_start_indices = None

        self._initialize(method=method)

        return

    def _initialize(self, kmer_filters=[], method: str = "single_pass"):
        """
        Initialize Kmers instance and populate internal kmer_sba_start_indices array.

        Args:
            kmer_filters (list, optional): _description_. Defaults to [].
            method (str, optional): which method to use for initialization.  "single_pass" runs
                faster, but temporarily uses more RAM (up to 2x).  "double_pass" runs slower (~2x)
                but uses less memory (as much as half).  Defaults to "single_pass".

        Raises:
            ValueError: method not recognized
        """
        if kmer_filters != []:
            raise NotImplementedError("kmer_filters have not been implemented")

        if method == "double_pass":
            # TODO: "double_pass" implementation counts the number of kmers first, initializes an
            # array of the correct size, and then populates it on-the-fly. Requires less memory
            raise NotImplementedError(f"method '{method}' has not been implemented")
        elif method == "single_pass":
            self._initialize_single_pass(kmer_filters=kmer_filters)
        else:
            raise ValueError(f"method '{method}' not recognized")

        self._is_initialized = True

    def _initialize_single_pass(self, kmer_filters=[]):
        """
        Initialize Kmers in a single pass. This loads all unfiltered kmers into memory, filters
        them in place, and then creates a new array of length num_filtered_kmers. This runs
        faster than a "double pass" initialization, but temporarily uses more memory (up to 2x).

        Args:
            kmer_filters (list, optional): _description_. Defaults to [].

        Raises:
            AssertionError: logic check
        """

        if kmer_filters != []:
            raise NotImplementedError("kmer_filters have not been implemented")

        num_kmers = self._get_unfiltered_kmer_count()
        if num_kmers > 2**32 - 1:
            msg = "the size of the required kmers array exceeds the limit set by a uint32"
            raise NotImplementedError(msg)

        # initialize array large enough to hold all unfiltered kmers
        self.kmer_sba_start_indices = np.zeros(num_kmers, dtype=np.uint32)

        # iterate over records initializing kmer start indices
        record_generator = self.sequence_collection.iter_records()
        last_filled_index = -1
        for _, sba_seg_start_idx, sba_seg_end_idx in record_generator:
            num_kmers_in_record = (sba_seg_end_idx - sba_seg_start_idx + 1) - self.min_kmer_len + 1

            kmer_start = last_filled_index + 1
            kmer_end = last_filled_index + 1 + num_kmers_in_record
            sba_start = sba_seg_start_idx
            sba_end = sba_seg_end_idx + 1 - self.min_kmer_len + 1
            self.kmer_sba_start_indices[kmer_start:kmer_end] = np.arange(
                sba_start, sba_end, dtype=np.uint32
            )
            last_filled_index += num_kmers_in_record

        if last_filled_index != num_kmers - 1:
            raise AssertionError(
                f"logic error: last_filled_index ({last_filled_index}) != num_kmers - 1 ({num_kmers - 1})"
            )

        # TODO: next step is to filter kmers in place

        return

    def _get_unfiltered_kmer_count(self) -> int:
        """
        Calculate the number of unfiltered kmers and the total length of the kmer array for the
        loaded sequence_collection.

        Raises:
            ValueError: empty sequence_collection

        Returns:
            int: num_kmers
        """
        # TODO: when SequenceCollection has a method to get num_records, remove the counter below
        num_kmers = 0
        num_records = 0
        record_generator = self.sequence_collection.iter_records()
        for _, sba_seg_start_idx, sba_seg_end_idx in record_generator:
            num_kmers_in_record = (sba_seg_end_idx - sba_seg_start_idx + 1) - self.min_kmer_len + 1
            num_kmers += num_kmers_in_record
            num_records += 1

        # TODO: when SequenceCollection has a method to get num_records, move this to top of func
        if num_records == 0:
            raise ValueError(f"SequenceCollection does not have any records")

        return num_kmers

    def __len__(self):
        pass

    def __getitem__(self):
        pass

    def kmer_generator(self, kmer_len, fields=["kmer"], unique_only=False):
        """
        Defines a generator that yields tuples with the requested information about
        each kmer

        Args:
            kmer_len: size of kmer to yield

        Allowed Fields:
            kmer_num
            kmer
            reverse_complement_kmer
            canonical_kmer
            is_canonical_kmer
            record_num
            record_name
            record_seq_forward_start_idx
            record_seq_forward_end_idx
            kmer_count
            NOTE: only allowed if unique_only is True

        NOTE: I'm worried about performance, I would recommend a simple initial
        implementation and profiling
        """

        pass

    def save_state(self, save_file_base, include_sequence_collection):
        """
        Save current state to file so that it can be reloaded from disk.  Consider some
        sort of a check on the sequence
        file to ensure reloading from a fasta works.
        NOTE: this is a medium-to-large task to do properly with saved metadata
        """
        pass

    def load_state(self, save_file_base, sequence_collection=None):
        """
        Load state from file
        """
        pass

    def unique(self, kmer_len, inplace=False):
        """
        Discard repeated kmers keep only unique kmers.
        """
        pass

    def count_unique(self, kmer_len):
        """
        Count the number of unique kmers
        """
        pass

    def sort(self):
        """
        Sort kmers.  This is the most computationally expensive step
        """
        pass

    def to_csv(self, kmer_len, output_file_path, fields=["kmer"]):
        """
        Write all kmers to CSV file using a simple function.
        """
        pass
