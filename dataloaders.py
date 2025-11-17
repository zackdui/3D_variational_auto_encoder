import torch
from torch.utils.data import Dataset, DataLoader

class RandomPatch3DDataset(Dataset):
    def __init__(self, base_dataset: Dataset, patch_size, patches_per_volume: int = 1, dataset_length: int | None =None):
        """
        base_dataset: a Dataset where __getitem__(i) returns a volume (C, D, H, W) or a dict with a volume.
        patch_size: (pd, ph, pw)
        dataset_length: optional, if you want to repeat volumes mroe times with different random patches
            Minimum dataset length is the length of the volumes
        patches_per_volume: Number of patches to return per volume
        """
        
        self.base_dataset = base_dataset
        self.patch_size = patch_size  # (pd, ph, pw)
        default_len = len(base_dataset) * patches_per_volume
        if dataset_length is None:
            self.dataset_length = default_len
        else:
            # allow making this arbitrarily large; we’ll wrap indices
            self.dataset_length = max(default_len, dataset_length)
        self.patches_per_volume = patches_per_volume

    def __len__(self):
        return self.dataset_length
    
    def _extract_volume(self, item):
        """
        In case your base dataset returns dicts like {"image": vol, "label": ...},
        adapt this accordingly.
        """
        if isinstance(item, torch.Tensor):
            return item  # assume (C, D, H, W)
        if isinstance(item, dict):
            # adjust this key name to match your base dataset
            return item["volume"] if "volume" in item else item["image"]
        raise TypeError(f"Unexpected item type from base_dataset: {type(item)}")

    def _random_crop_3d(self, vol):
        C, D, H, W = vol.shape
        pd, ph, pw = self.patch_size

        # handle smaller volumes by clamping
        D_max = max(D - pd, 0)
        H_max = max(H - ph, 0)
        W_max = max(W - pw, 0)

        z0 = 0 if D_max == 0 else torch.randint(0, D_max + 1, (1,)).item()
        y0 = 0 if H_max == 0 else torch.randint(0, H_max + 1, (1,)).item()
        x0 = 0 if W_max == 0 else torch.randint(0, W_max + 1, (1,)).item()

        z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw

        vol_patch = vol[:, z0:z1, y0:y1, x0:x1]
        return vol_patch

    def __getitem__(self, idx):
        # map global idx -> which volume we’re drawing a patch from
        n_vols = len(self.base_dataset)

        # each volume contributes `patches_per_volume` items
        vol_idx = (idx // self.patches_per_volume) % n_vols

        base_item = self.base_dataset[vol_idx]
        vol = self._extract_volume(base_item)

        patch = self._random_crop_3d(vol)  # (C, pd, ph, pw)
        return patch

def _compute_starts(full, patch, stride):
    """
    full: full size in one dimension (e.g., D, H, or W)
    patch: patch size in that dimension
    stride: stride in that dimension

    returns: list of starting indices that cover [0, full)
    """
    if full <= patch:
        # Volume smaller than patch: just start at 0
        return [0]

    starts = list(range(0, full - patch + 1, stride))
    # Ensure we cover the end exactly
    last_start = full - patch
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def make_grid_patches_3d(volume, patch_size, stride):
    """
    volume: (C, D, H, W)
    patch_size: (pd, ph, pw)
    stride: (sd, sh, sw)
    returns: list of (patch, (z0, y0, x0)) tuples
    """
    C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    z_starts = _compute_starts(D, pd, sd)
    y_starts = _compute_starts(H, ph, sh)
    x_starts = _compute_starts(W, pw, sw)

    patches = []
    for z0 in z_starts:
        z1 = z0 + pd
        for y0 in y_starts:
            y1 = y0 + ph
            for x0 in x_starts:
                x1 = x0 + pw
                patch = volume[:, z0:z1, y0:y1, x0:x1]
                patches.append((patch, (z0, y0, x0)))

    return patches


class AllPatch3DDataset(Dataset):
    def __init__(self, base_dataset, patch_size, stride):
        """
        For this datset num_workers should be 0 and shuffle should be False
        This Dataset will take every patch from every volume
        Volumes should be the same size for proper dataset length evaluation
        base_dataset: yields volumes of shape (C, D, H, W)
        patch_size: (pd, ph, pw)
        stride: (sd, sh, sw)
        """
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.stride = stride
        self.cur_patches = []
        self.num_patches = len(
            make_grid_patches_3d(self.base_dataset[0], self.patch_size, self.stride)
        )
        self.cur_volume_idx = 0

    def __len__(self):
        return len(self.base_dataset) * self.num_patches
    
    def __getitem__(self, idx):
        if len(self.cur_patches) == 0:
            vol = self.base_dataset[self.cur_volume_idx]
            self.cur_volume_idx += 1
            self.cur_patches = make_grid_patches_3d(vol, self.patch_size, self.stride)
        return self.cur_patches.pop()[0]

class RandomVolumeDataset(Dataset):
    def __init__(self, num_volumes, shape=(1, 208, 512, 512)):
        """
        num_volumes: how many random volumes to generate
        shape: (C, D, H, W)
        """
        self.num_volumes = num_volumes
        self.shape = shape

    def __len__(self):
        return self.num_volumes

    def __getitem__(self, idx):
        # return a random float32 tensor shaped like a CT volume
        return torch.rand(self.shape)

