import argparse
import logging
import time

from profiling.utils import (
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

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--small", action="store_true", help="Run the small test set")
    group.add_argument("--medium", action="store_true", help="Run the medium test set")
    group.add_argument("--large", action="store_true", help="Run the full test set")
    parser.add_argument(
        "--num-iter",
        "-n",
        type=int,
        default=3,
        help="number of run times over which to average.  Must be > 0.",
    )

    args = parser.parse_args()

    if args.small:
        args.run_size = "small"
    elif args.large:
        args.run_size = "large"
    else:
        args.run_size = "medium"

    if args.num_iter < 1:
        raise ValueError(f"number of iterations requested ({args.num_iter}) is < 1")

    return args


def main():
    """
    Run profiling script
    """
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    num_iterations = args.num_iter

    ###########################
    # seq_list init profiling #
    ###########################
    if args.run_size == "small":
        total_seq_len_list = [1e3, 1e4, 1e5]
    elif args.run_size == "medium":
        total_seq_len_list = [1e3, 1e5, 1e7]
    elif args.run_size == "large":
        total_seq_len_list = [1e4, 1e5, 1e6, 1e7, 1e8]
    else:
        raise ValueError(f"run_size ({args.run_size}) not recognized")

    start_time = time.time()
    logging.info("sequence list init: 'forward' strand")
    df = profile_seq_list_init(
        total_seq_len_list=total_seq_len_list,
        strand_to_load="forward",
        num_iterations=num_iterations,
        seed=42,
        discard_first_run=True,
    )
    logging.info(f"\n{df}")

    logging.info("sequence list init: 'reverse_complement' strand")
    df = profile_seq_list_init(
        total_seq_len_list=total_seq_len_list,
        strand_to_load="reverse_complement",
        num_iterations=num_iterations,
        seed=42,
        discard_first_run=True,
    )
    logging.info(f"\n{df}")

    logging.info("sequence list init: 'both' strand")
    df = profile_seq_list_init(
        total_seq_len_list=total_seq_len_list,
        strand_to_load="both",
        num_iterations=num_iterations,
        seed=42,
        discard_first_run=True,
    )
    logging.info(f"\n{df}")
    run_time_init_profiling = time.time() - start_time

    ################################
    # get segment number profiling #
    ################################
    if args.run_size == "small":
        num_chromosomes_list = [1, 10, 100]
    elif args.run_size == "medium":
        num_chromosomes_list = [1, 100, 10000]
    elif args.run_size == "large":
        num_chromosomes_list = [1, 10, 100, 1000, int(1e4), int(1e5), int(1e6)]
    else:
        raise ValueError(f"run_size ({args.run_size}) not recognized")

    start_time = time.time()
    logging.info("get_segment_num_from_sba_index: 'forward' strand")
    df = profile_get_segment_num_from_sba_index(
        num_chromosomes_list=num_chromosomes_list,
        total_seq_len=int(1e6),
        strand_to_load="forward",
        num_iterations=num_iterations,
        num_lookups=int(1e4),
        seed=42,
        discard_first_run=True,
    )
    logging.info(f"\n{df}")
    run_time_get_seg_num = time.time() - start_time

    logging.info("get_segment_num_from_sba_index: 'forward' strand")
    df = profile_get_segment_num_from_sba_index(
        num_chromosomes_list=num_chromosomes_list,
        total_seq_len=int(1e6),
        strand_to_load="reverse_complement",
        num_iterations=num_iterations,
        num_lookups=int(1e4),
        seed=42,
        discard_first_run=True,
    )
    logging.info(f"\n{df}")
    run_time_get_seg_num = time.time() - start_time

    logging.info("get_segment_num_from_sba_index: 'both' strand")
    df = profile_get_segment_num_from_sba_index(
        num_chromosomes_list=num_chromosomes_list,
        total_seq_len=int(1e6),
        strand_to_load="both",
        num_iterations=num_iterations,
        num_lookups=int(1e4),
        seed=42,
        discard_first_run=True,
    )
    logging.info(f"\n{df}")
    run_time_get_seg_num = time.time() - start_time

    # report total run time of this profiling function
    logging.info(f"Total init profiling run time: {run_time_init_profiling}")
    logging.info(f"Total get_segment_num_from_sba_index run time: {run_time_get_seg_num}")

    return


if __name__ == "__main__":
    main()
