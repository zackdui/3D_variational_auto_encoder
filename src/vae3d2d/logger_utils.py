import os
import logging
from pathlib import Path
import torch

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

class LossMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.recon_sum = 0.0
        self.kl_sum = 0.0
        self.beta_kl_sum = 0.0
        self.total_sum = 0.0
        self.n = 0

    @torch.no_grad()
    def update(self, recon, kl, beta):
        r = float(recon.detach())
        k = float(kl.detach())
        bk = float(beta) * k
        self.recon_sum += r
        self.kl_sum += k
        self.beta_kl_sum += bk
        self.total_sum += (r + bk)
        self.n += 1

    def averages(self):
        if self.n == 0:
            return {}
        recon = self.recon_sum / self.n
        beta_kl = self.beta_kl_sum / self.n
        return {
            "avg/recon": recon,
            "avg/kl": self.kl_sum / self.n,
            "avg/beta_kl": beta_kl,
            "avg/total": self.total_sum / self.n,
            "avg/beta_kl_over_recon": beta_kl / (recon + 1e-12),
        }