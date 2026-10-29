import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="ostrack", choices=["ostrack"])
    parser.add_argument("--config", default="strf_vitb_256")
    parser.add_argument("--save_dir", default="./output")
    parser.add_argument("--mode", choices=["single", "multiple"], default="single")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lmdb", type=int, choices=[0, 1], default=0)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--data_root", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable]
    if args.mode == "multiple":
        if args.nproc_per_node < 1:
            raise ValueError("nproc_per_node must be positive.")
        command += [
            "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(args.nproc_per_node),
        ]
    command += [
        str(root / "lib/train/run_training.py"),
        "--script", args.script,
        "--config", args.config,
        "--save_dir", str(Path(args.save_dir).expanduser().resolve()),
        "--seed", str(args.seed),
        "--use_lmdb", str(args.use_lmdb),
    ]
    if args.pretrained is not None:
        command += ["--pretrained", str(Path(args.pretrained).expanduser().resolve())]
    if args.data_root is not None:
        command += ["--data_root", str(Path(args.data_root).expanduser().resolve())]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
