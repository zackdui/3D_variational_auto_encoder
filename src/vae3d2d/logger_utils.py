import os
import logging
from pathlib import Path

def setup_logger(save_dir: str = None, name: str = "vae_trainer", train=True) -> logging.Logger:
    if save_dir is None:
        cwd = os.getcwd()              # get current working directory
        save_path = Path(os.path.join(cwd, "logging", name))
    else:
        save_path = Path(os.path.join(save_dir, name))
        # save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if train:
        log_file = save_path / "train.log"
    else:
        log_file = save_path / "eval.log"

    logging.basicConfig(
        level=logging.INFO,  # log INFO and above
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),          # → console
            logging.FileHandler(log_file),    # → train.log
        ],
    )
    logger = logging.getLogger(name)
    logger.info("Logger initialized. Writing logs to %s", log_file)
    return logger