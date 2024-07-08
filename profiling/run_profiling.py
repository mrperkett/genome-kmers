import argparse
import logging
import os
import time

from profiling.utils import (
    profile_fasta_init,
    profile_get_segment_num_from_sba_index,
    profile_seq_list_init,
)


def parse_args():
    """
    Parse and validate command line arguments.

    Returns:
        argparse.Namespace: arguments are stored as member variables
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-size",
        "-s",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="size of the profiling run.  'small' is quick, but incomplete.  'large' is slower, but more complete.",
    )
    parser.add_argument(
        "--num-iter",
        "-n",
        type=int,
        default=3,
        help="number of run times over which to average.  Must be > 0.",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default="all",
        choices=["all", "seq_list_init", "fasta_init", "get_segment_num"],
    )
    parser.add_argument(
        "--output-base",
        "-o",
        type=str,
        default=None,
        help="output file base that is used as the prefix for profiling output files",
    )

    args = parser.parse_args()

    if args.num_iter < 1:
        raise ValueError(f"number of iterations requested ({args.num_iter}) is < 1")

    # verify that the output base is valid
    if args.output_base is not None:
        dir_path, file_base = os.path.split(args.output_base)
        if dir_path != "" and not os.path.isdir(dir_path):
            raise ValueError(f"directory path ({dir_path}) does not exist")
        if len(file_base) < 1:
            raise ValueError(
                f"a file base is not present in the output_base argument ({args.output_base})"
            )

    return args


def run_seq_list_init_profiling(run_size: str, num_iterations: int, output_base: str = None):
    """
    Profile SequenceCollection seq_list initialization.

    Args:
        run_size (str): "small", "medium", or "large".  Determines level of profiling with
            "small" giving a quick but incomplete profiling summary and "large" giving a slower
            but more complete profiling summary.
        num_iterations (int): number of iterations to average over during profiling
        output_base (str, optional): output file base that is used as the prefix for profiling
            output files.  If not provided, then no files are written.  Defaults to None.

    Raises:
        ValueError: raised if run_size is not recognized
    """
    if run_size == "small":
        total_seq_len_list = [1e3, 1e4, 1e5]
    elif run_size == "medium":
        total_seq_len_list = [1e3, 1e5, 1e7]
    elif run_size == "large":
        total_seq_len_list = [1e4, 1e5, 1e6, 1e7, 1e8]
    else:
        raise ValueError(f"run_size ({run_size}) not recognized")

    start_time = time.time()

    for strand_to_load in ("forward", "reverse_complement", "both"):
        logging.info(f"sequence list init: '{strand_to_load}' strand")
        df = profile_seq_list_init(
            total_seq_len_list=total_seq_len_list,
            strand_to_load=strand_to_load,
            num_iterations=num_iterations,
            seed=42,
            discard_first_run=True,
        )
        logging.info(f"\n{df}")

        if output_base is not None:
            out_file_path = f"{output_base}-seq-list-init-{strand_to_load}.csv"
            df.to_csv(out_file_path, header=True, index=False)
            logging.info(f"profiling info written to '{out_file_path}'")

    run_time_init_profiling = time.time() - start_time
    logging.info(f"Total init profiling run time: {run_time_init_profiling}")


def run_get_segment_num_profiling(run_size: str, num_iterations: int, output_base: str = None):
    """
    Profile SequenceCollection get_segment_num_from_sba_index, which is used to determine to which
    chromosome a given kmer belongs.

    Args:
        run_size (str): "small", "medium", or "large".  Determines level of profiling with
            "small" giving a quick but incomplete profiling summary and "large" giving a slower
            but more complete profiling summary.
        num_iterations (int): number of iterations to average over during profiling
        output_base (str, optional): output file base that is used as the prefix for profiling
            output files.  If not provided, then no files are written.  Defaults to None.

    Raises:
        ValueError: raised if run_size is not recognized
    """
    if run_size == "small":
        num_chromosomes_list = [1, 10, 100]
    elif run_size == "medium":
        num_chromosomes_list = [1, 100, 10000]
    elif run_size == "large":
        num_chromosomes_list = [1, 10, 100, 1000, int(1e4), int(1e5), int(1e6)]
    else:
        raise ValueError(f"run_size ({run_size}) not recognized")

    start_time = time.time()
    for strand_to_load in ("forward", "reverse_complement", "both"):
        logging.info(f"get_segment_num_from_sba_index: '{strand_to_load}' strand")
        df = profile_get_segment_num_from_sba_index(
            num_chromosomes_list=num_chromosomes_list,
            total_seq_len=int(1e6),
            strand_to_load=strand_to_load,
            num_iterations=num_iterations,
            num_lookups=int(1e4),
            seed=42,
            discard_first_run=True,
        )
        logging.info(f"\n{df}")

        if output_base is not None:
            out_file_path = f"{output_base}-segment-num-from-sba-index-{strand_to_load}.csv"
            df.to_csv(out_file_path, header=True, index=False)
            logging.info(f"profiling info written to '{out_file_path}'")

    run_time_get_seg_num = time.time() - start_time
    logging.info(f"Total get_segment_num_from_sba_index run time: {run_time_get_seg_num}")


def run_fasta_init_profiling(run_size: str, num_iterations: int, output_base: str = None):
    """
    Profile SequenceCollection initialization by fasta file.

    Args:
        run_size (str): "small", "medium", or "large".  Determines level of profiling with
            "small" giving a quick but incomplete profiling summary and "large" giving a slower
            but more complete profiling summary.
        num_iterations (int): number of iterations to average over during profiling
        output_base (str, optional): output file base that is used as the prefix for profiling
            output files.  If not provided, then no files are written.  Defaults to None.

    Raises:
        ValueError: raised if run_size is not recognized
    """
    if run_size == "small":
        total_seq_lengths = [int(1e3), int(1e4), int(1e5)]
    elif run_size == "medium":
        total_seq_lengths = [int(1e5), int(1e6), int(1e7)]
    elif run_size == "large":
        total_seq_lengths = [int(1e4), int(1e5), int(1e6), int(1e7), int(1e8)]
    else:
        raise ValueError(f"run_size ({run_size}) not recognized")

    start_time = time.time()
    logging.info(f"profile_fasta_init")
    df = profile_fasta_init(
        total_seq_lengths=total_seq_lengths,
        num_chromosomes=10,
        max_line_length=80,
        strand="forward",
        num_iterations=num_iterations,
        seed=42,
        discard_first_run=True,
    )
    logging.info(f"\n{df}")

    if output_base is not None:
        out_file_path = f"{output_base}-fasta-init.csv"
        df.to_csv(out_file_path, header=True, index=False)
        logging.info(f"profiling info written to '{out_file_path}'")

    run_time = time.time() - start_time
    logging.info(f"Total fasta_init_profiling run time: {run_time}")


def main():
    """
    Run profiling script
    """
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    num_iterations = args.num_iter
    run_size = args.run_size

    if args.category in ("all", "seq_list_init"):
        run_seq_list_init_profiling(run_size, num_iterations, args.output_base)

    if args.category in ("all", "get_segment_num"):
        run_get_segment_num_profiling(run_size, num_iterations, args.output_base)

    if args.category in ("all", "fasta_init"):
        run_fasta_init_profiling(run_size, num_iterations, args.output_base)

    return


if __name__ == "__main__":
    main()
