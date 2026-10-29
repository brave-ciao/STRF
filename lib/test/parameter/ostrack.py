from copy import deepcopy
from pathlib import Path

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.utils import TrackerParams


def parameters(yaml_name: str, run_id=None, checkpoint=None):
    env = env_settings()
    config = deepcopy(cfg)
    yaml_file = Path(env.prj_dir) / "experiments" / "ostrack" / f"{yaml_name}.yaml"
    update_config_from_file(str(yaml_file), base_cfg=config)
    selected = checkpoint or config.TEST.CHECKPOINT
    epoch = run_id if run_id is not None else config.TEST.EPOCH
    if not selected:
        if epoch is None:
            raise ValueError("Provide --checkpoint or --epoch, or set TEST.CHECKPOINT in the YAML.")
        selected = (Path(env.save_dir) / "checkpoints/train/ostrack" /
                    yaml_name / f"OSTrack_ep{int(epoch):04d}.pth.tar")
    selected = Path(selected).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(selected)

    params = TrackerParams()
    params.cfg = config
    params.checkpoint = str(selected)
    params.template_factor = config.TEST.TEMPLATE_FACTOR
    params.template_size = config.TEST.TEMPLATE_SIZE
    params.search_factor = config.TEST.SEARCH_FACTOR
    params.search_size = config.TEST.SEARCH_SIZE
    params.save_all_boxes = False
    return params
