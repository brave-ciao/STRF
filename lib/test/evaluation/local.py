import os
from pathlib import Path

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()
    root = Path(__file__).resolve().parents[3]
    settings.prj_dir = str(root)
    settings.save_dir = os.environ.get("STRF_OUTPUT_DIR", str(root / "output"))
    settings.results_path = str(Path(settings.save_dir) / "test/tracking_results")
    settings.network_path = ""
    settings.result_plot_path = ""
    settings.segmentation_path = ""
    settings.lsotb_path = os.environ.get("STRF_LSOTB_TEST", "")
    settings.ptbtir_path = os.environ.get("STRF_PTBTIR_TEST", "")
    settings.lashertir_path = os.environ.get("STRF_LASHERTIR_TEST", "")
    settings.davis_dir = ""
    settings.got10k_lmdb_path = ""
    settings.got10k_path = ""
    settings.got_packed_results_path = ""
    settings.got_reports_path = ""
    settings.itb_path = ""
    settings.lasot_extension_subset_path = ""
    settings.lasot_extension_subset_path_path = ""
    settings.lasot_lmdb_path = ""
    settings.lasot_path = ""
    settings.nfs_path = ""
    settings.otb_path = ""
    settings.tc128_path = ""
    settings.tn_packed_results_path = ""
    settings.tnl2k_path = ""
    settings.tpl_path = ""
    settings.trackingnet_path = ""
    settings.uav_path = ""
    settings.vot18_path = ""
    settings.vot20_path = ""
    settings.vot22_path = ""
    settings.vot_path = ""
    settings.youtubevos_dir = ""
    return settings
