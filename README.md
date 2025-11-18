# 3D Variational Autoencoder Framework

This repository provides an easy-to-use framework for training 3D Variational Autoencoders (VAEs), with a focus on grayscale medical imaging data such as CT or MRI volumes. The codebase is fully compatible with 2D images as well, making it applicable beyond volumetric data.

The framework includes flexible data loaders, a modular VAE architecture, patch-based training utilities, mixed-precision support, optional distributed training (DDP), and seamless Weights & Biases logging.

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
