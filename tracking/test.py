import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_tracker(tracker_name, tracker_param, checkpoint=None, epoch=None,
                dataset_name="lsotb", sequence=None, threads=0, num_gpus=1):
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.running import run_dataset
    from lib.test.evaluation.tracker import Tracker

    tracker = Tracker(tracker_name, tracker_param, dataset_name, run_id=epoch,
                      checkpoint=checkpoint)
    tracker.get_parameters()
    dataset = get_dataset(dataset_name)
    if sequence is not None:
        try:
            sequence = int(sequence)
        except ValueError:
            pass
        dataset = [dataset[sequence]]
    run_dataset(dataset, [tracker], debug=False, threads=threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker_name", default="ostrack", choices=["ostrack"])
    parser.add_argument("--tracker_param", default="strf_vitb_256")
    parser.add_argument("--dataset_name", "--dataset", default="lsotb")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--checkpoint", default=None)
    selection.add_argument("--epoch", "--runid", dest="epoch", type=int, default=None)
    args = parser.parse_args()
    if args.threads < 0 or args.num_gpus < 1:
        parser.error("threads must be nonnegative and num_gpus must be positive.")
    run_tracker(args.tracker_name, args.tracker_param, checkpoint=args.checkpoint,
                epoch=args.epoch, dataset_name=args.dataset_name,
                sequence=args.sequence, threads=args.threads, num_gpus=args.num_gpus)


if __name__ == "__main__":
    main()
