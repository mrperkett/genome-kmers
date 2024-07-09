class Kmers:
    """
    Class that holds pointers to the start of each kmer location and references a
    sequence_collection.
    """

    def __init__(
        self,
        sequence_collection=None,
        min_kmer_len=None,
        max_kmer_len=None,
        max_ambiguous_bases=5,
        source_strand=None,
        track_strands_separately=None,
    ):
        """
        Args:
            sequence_collection:
                NOTE: in early implementations, it probably makes sense to require
                certain strands to be loaded rather than going through difficult
                implementation gymnastics.
            max_kmer_len:
            max_ambiguous_bases:
            source_strand: user specifies which strand to use as a source of kmers
                ("forward", "reverse_complement", "both")
            track_strands_separately: if source_strand == "both", then you must specify
                this variable as True/False.  True will treat kmers on each strand
                separately.  False will not distinguish between strands when checking
                for kmer equality.
        """
        self.sequence_collection = sequence_collection
        self.max_kmer_len = max_kmer_len
        self.min_kmer_len = min_kmer_len
        self.kmer_source_strand = source_strand
        self.track_strands_separately = track_strands_separately
        self._initialized = False
        self._is_set = False
        self._is_sorted = False
        return

    def initialize(self, kmer_filters=[]):
        """
        Initialize the forward and/or reverse_complement sequence byte arrays that hold
        a pointer to each
        NOTE: need to account for min_kmer_len, max_kmer_len, and
        **max_ambiguous_bases**.  max_ambiguous bases gets tricky
        implementation-wise.  Maybe defer this to later after profiling?
        Could also use trick of replacing Ns with random bases?
        Could just avoid the large N rafts and assume it will be good enough
        behavior without further modification?
        Identify all ranges for ambiguous bases.  Iterate over start indices
        that do not immediately violate ambiguous base count.  Kmer is defined
        to end when abiguous base count is met or end of seq is found.
        NOTE: this may need to be sped up using nb.jit.  I'm not sure how tricky this
        will be with filters
        NOTE: recommend starting not implementing filters, and then implementing
        filters to see how it impacts performance
        """
        pass

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
