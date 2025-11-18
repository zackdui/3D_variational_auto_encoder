# vae3d2d

![PyPI](https://img.shields.io/pypi/v/vae3d2d)

---

# 3D Variational Autoencoder Short Description

This repository provides an easy-to-use framework for training 3D Variational Autoencoders (VAEs), with a focus on grayscale medical imaging data such as CT or MRI volumes. The codebase is fully compatible with 2D images as well, making it applicable beyond volumetric data.

The framework includes flexible data loaders, a modular VAE architecture, patch-based training utilities, mixed-precision support, optional distributed training (DDP), and seamless Weights & Biases logging.

--

# Quick Start

You can use this project in **two ways**:

1. **Clone the repository** (for development or customization)  
2. **Install it as a PyPI package** (for direct use)

Both methods support CPU or GPU machines.

---

## Requirements

- **Python 3.9+**
- **PyTorch 2.0+** (CPU or CUDA, depending on your machine)
- GPU optional but recommended for 3D workloads  
- Install either:
  - `requirements-gpu.txt` → includes CUDA-enabled PyTorch  
  - `requirements.txt` → CPU environment (PyTorch must be installed first)  
  - `requirements.in` → fully pinned development environment  

---

## Option 1 — Clone the Repository

### 1. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Choose depending on your machine:

- **GPU machine** (CUDA)  
  → `requirements-gpu.txt` already includes the correct CUDA-enabled PyTorch  
  ```bash
  pip install -r requirements-gpu.txt
  ```

- **CPU machine**  
  → Install PyTorch manually *first* (select command from https://pytorch.org/get-started/locally), e.g.  
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
  Then install the remaining dependencies:
  ```bash
  pip install -r requirements.txt
  ```

- **Full development environment (pinned)**  
  ```bash
  pip install -r requirements.in
  ```

---

## Option 2 — Install via PyPI

### 1. Install PyTorch **first**
(Choose CPU or CUDA version from https://pytorch.org/get-started/locally.)

Example (CPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install the package
```bash
pip install vae3d2d
```

---

# 3D Variational Autoencoder Framework

---

## Data Loaders

The provided dataloaders allow you to take **any 3D image dataset** and:

- **Extract patches** from volumes to train your model on smaller subregions instead of full 3D images.
- Generate overlapping or non-overlapping patches.
- Randomly sample patches per volume each epoch.
- Use the test loader to output any number of random patches of any chosen shape, useful for visualization or debugging.

This makes it straightforward to train large 3D models even on limited GPU memory.

---

## Model

`CustomVAE` (defined in `model.py`) is a highly configurable 3D Variational Autoencoder. It supports:

- User-defined **downsampling and upsampling blocks**.
- Optional **attention** in the bottleneck:
  - Global attention
  - Patch attention
  - Relative position bias (optional)
- Flexible latent dimensionality
- Multiple normalization and activation options
- Support for both 2D and 3D inputs

The architecture is modular, making it easy to extend or customize for research.

---

## Training

Training utilities come with:

- Automatic support for **DistributedDataParallel (DDP)**  
  (DDP is initialized automatically when launched via `torchrun`).
- Optional **mixed precision (AMP)** for memory-efficient training.
- **Gradient accumulation** for large effective batch sizes.
- **W&B logging** for losses, reconstructions, and sample visualizations.
- Automatic checkpointing (periodic and best-model).

### Full-Volume Training

To train directly on full 3D images, simply call the `train_3d` function with:

```
patching = None
```

The model receives entire volumes as input. This is the simplest way to train if your GPU memory allows it.

### Patch-Based Training

Patch-based training is available by setting:

- `patching = "full"`  
  → each volume is fully decomposed into patches; the model trains on **all** patches.

- `patching = "random_parts"`  
  → a fixed number of random patches are sampled per volume every epoch, reducing computation and adding useful stochasticity.

Patch training enables high-resolution 3D models to run on modest hardware.

---

## Evaluation

Evaluation supports both full-volume and patch-based reconstruction:

- `mode="full"`  
  → the entire 3D volume is fed through the model.

- `mode="patch"`  
  → the volume is broken into overlapping patches, reconstructed patch-by-patch, and stitched back together using blending or averaging.

The final reconstruction loss is computed on the fully reassembled volume.

W&B logging is also supported during evaluation, including sample reconstructions and GIF/MP4 visualizations.

---

## Example: Minimal Training & Evaluation Script

Below is a compact, end-to-end example showing how to train and evaluate a 3D VAE using this library.  
Save this as `main.py`:

```python
import torch
from vae3d2d import (
    CustomVAE,
    AttnParams,
    RandomVolumeDataset,
    training_3D,
    eval_model_3D,
    setup_logger,
)

if __name__ == "__main__":
    # 1) Create a toy 3D dataset
    train_dataset = RandomVolumeDataset(20, shape=(1, 64, 128, 128))

    # 2) Define model + attention
    attn = AttnParams(
        num_heads=4,
        dim_head=32,
        dropout=0.1,
        window_size=(4, 8, 8),
        use_rel_pos_bias=True
    )

    model = CustomVAE(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        use_attn=True,
        attn_params=attn,
        vae_use_log_var=True,
    )

    # 3) Train with random patch sampling
    logger = setup_logger(name="example_train_logs")

    training_3D(
        model,
        train_dataset,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=1e-4, weight_decay=1e-5),
        epochs=3,
        accum_steps=4,
        patching="random_parts",        # or None for full-volume training
        model_file_name="example.pt",
        logger=logger,
        use_wandb=False,
    )

    # 4) Evaluate on separate random test volumes
    test_dataset = RandomVolumeDataset(5, shape=(1, 208, 512, 512))

    eval_model_3D(
        model,
        test_dataset,
        batch_size=1,
        mode="patch",                   # "full" or "patch"
        patch_size=(64, 128, 128),
        stride=(32, 64, 64),
        save_dir="./example_eval_outputs",
        use_blending=True,
        logger=logger,
        use_wandb=False,
    )
```

### Multi-GPU (DDP) Training

To train on multiple GPUs, launch the same script with:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py
```

Or for 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py
```

# Build & Publish

## Rebuild the package

1. Bump the version in:
   - `pyproject.toml`
   - `vae3d2d/__init__.py`

2. Clean old builds:
   ```bash
   rm -rf dist build *.egg-info
   ```

3. Build:
   ```bash
   python -m build
   ```

4. Upload:
   ```bash
   twine upload dist/*
   ```

Make sure you have the required tools:

```bash
pip install build twine
```

