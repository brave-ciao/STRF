import argparse
import importlib
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def init_seeds(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(script_name, config_name, local_rank=-1, save_dir="./output",
                 base_seed=42, use_lmdb=False, pretrained=None, data_root=None):
    import cv2
    from lib.train.admin.settings import Settings

    cv2.setNumThreads(0)
    init_seeds(base_seed + max(local_rank, 0))
    settings = Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = f"train/{script_name}/{config_name}"
    settings.local_rank = local_rank
    settings.save_dir = str(Path(save_dir).expanduser().resolve())
    settings.env.workspace_dir = settings.save_dir
    settings.use_lmdb = use_lmdb
    settings.pretrained = pretrained
    if data_root is not None:
        settings.env.lsotb_dir = data_root
    settings.cfg_file = str(ROOT / "experiments" / script_name / f"{config_name}.yaml")
    importlib.import_module(f"lib.train.train_script").run(settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="ostrack", choices=["ostrack"])
    parser.add_argument("--config", default="strf_vitb_256")
    parser.add_argument("--save_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lmdb", type=int, choices=[0, 1], default=0)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--local_rank", "--local-rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", -1)))
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
    else:
        torch.cuda.set_device(0)
    try:
        run_training(args.script, args.config, local_rank=args.local_rank,
                     save_dir=args.save_dir, base_seed=args.seed,
                     use_lmdb=bool(args.use_lmdb), pretrained=args.pretrained,
                     data_root=args.data_root)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
