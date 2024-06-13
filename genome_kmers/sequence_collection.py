class SequenceCollection:
    """
    Holds all the information contained within a fasta file in a format conducive to
    kmer sorting.  Each header and its corresponding sequence is called a record.
    """

    def __init__(
        self, fasta_file_path=None, sequence_list=None, strands_to_load="both"
    ):
        """
        fasta_file_path:
        sequence_list: [(header_0, seq_0), (..), ..]
        strand_to_load: "forward", "reverse_complement", "both"
        """
        self._strands_loaded = None
        self.forward_sba = None
        self.reverse_complement_sba = None
        self._record_forward_sba_start_indices = None
        self._record_reverse_complement_sba_start_indices = None
        self.record_names = None
        pass

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

    def _initialize_from_fasta(self, fasta_file_path):
        pass

    def _initialize_from_sequence_list(self, sequence_list):
        pass

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
    def _get_opposite_strand_sba_index(sba_idx, sba_len):
        """
        NOTE: this may need to be sped up using nb.jit.
        """
        pass

    @staticmethod
    def _get_opposite_strand_sba_start_indices(record_sba_start_indices, sba_len):
        """
        NOTE: this may need to be sped up using nb.jit.  Might be ok with np
        functionality though.
        """
        pass

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
