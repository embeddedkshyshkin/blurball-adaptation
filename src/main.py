import logging
import sys
from pathlib import Path

from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig

from runners import select_runner
from utils import mkdir_if_missing

log = logging.getLogger(__name__)


def _parse_positional_inputs():
    """Extract optional video/calibration positional arguments before Hydra parses CLI overrides."""
    args = sys.argv[1:]
    positional = [arg for arg in args if not arg.startswith("+") and "=" not in arg]
    overrides = [arg for arg in args if arg not in positional]

    if len(positional) > 2:
        raise SystemExit("Usage: python src/main.py [VIDEO] [CALIBRATION_JSON] [hydra overrides...]")

    if positional:
        overrides.append(f"input_vid={Path(positional[0]).resolve()}")
    if len(positional) == 2:
        overrides.append(f"calibration_file={Path(positional[1]).resolve()}")

    sys.argv = [sys.argv[0], *overrides]


@hydra.main(version_base=None, config_name='root', config_path='configs')
def main(cfg: DictConfig):
    if cfg['output_dir'] is None:
        cfg['output_dir'] = HydraConfig.get().run.dir
    mkdir_if_missing(cfg['output_dir'])

    runner = select_runner(cfg)
    runner.run()


if __name__ == "__main__":
    _parse_positional_inputs()
    main()
