import os
from pathlib import Path


class EnvironmentSettings:
    def __init__(self):
        root = Path(__file__).resolve().parents[3]
        self.workspace_dir = os.environ.get("STRF_OUTPUT_DIR", str(root / "output"))
        self.pretrained_networks = ""
        self.lsotb_dir = os.environ.get("STRF_LSOTB_TRAIN", "")
        self.lsotb_lmdb_dir = ""
        self.lasot_dir = ""
        self.lasot_lmdb_dir = ""
        self.got10k_dir = ""
        self.got10k_val_dir = ""
        self.got10k_lmdb_dir = ""
        self.trackingnet_dir = ""
        self.trackingnet_lmdb_dir = ""
        self.coco_dir = ""
        self.coco_lmdb_dir = ""
        self.lvis_dir = ""
        self.sbd_dir = ""
        self.imagenet_dir = ""
        self.imagenet_lmdb_dir = ""
        self.imagenetdet_dir = ""
        self.ecssd_dir = ""
        self.hkuis_dir = ""
        self.msra10k_dir = ""
        self.davis_dir = ""
        self.youtubevos_dir = ""
