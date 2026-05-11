import os
import torch
import h5py
import random
import numpy as np
import json
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from typing import Optional, Iterable, Literal, Generator

logger = logging.getLogger(__name__)

class StandardizeLabel:
	def __init__(self, mean: float = 0.0, std: float = 1e-05, clip: float = 5e-05):
		self.mean = mean
		self.std = std
		self.clip = clip

	def __call__(self, batch: list, training: bool = True) -> torch.Tensor:
		x, y = batch
		x = (torch.clamp(x, -self.clip, self.clip) - self.mean) / self.std
		return [x, y]


class V4ResampleTransform:
    """Resample 256 Hz → 128 Hz then clip+normalize for D4-v4 data.

    Input:  (C, 2560) float32 tensor at 256 Hz
    Output: (C, 1280) float32 tensor at 128 Hz, clipped and z-scored
    """
    _inner = StandardizeLabel()

    def __call__(self, batch):
        from scipy.signal import resample_poly
        x, y = batch
        x_np = resample_poly(x.numpy(), up=1, down=2, axis=-1).astype(np.float32)
        return self._inner((torch.from_numpy(x_np), y))

class H5PYDataset(Dataset):
    """Base class for loading data from multiple HDF5 files."""
    def __init__(self, path: str, key: str = "data"):
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} should be a directory")

        self.paths = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".hdf5")])
        self.key = key
        self.lengths = [self._get_file_length(p) for p in self.paths]
        self.cumulative_lengths = [0] + np.cumsum(self.lengths).tolist()

    def _get_file_length(self, path):
        with h5py.File(path, "r") as file:
            return file[self.key].shape[0]

    def _find_file_index(self, global_index):
        return np.searchsorted(self.cumulative_lengths, global_index, side='right') - 1

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, global_index: int):
        if global_index < 0 or global_index >= len(self):
            raise IndexError("Index out of bounds")
        file_index = self._find_file_index(global_index)
        local_index = global_index - self.cumulative_lengths[file_index]
        with h5py.File(self.paths[file_index], "r") as file:
            return torch.from_numpy(file[self.key][local_index])


class H5PYDatasetLabeled(H5PYDataset):
    """Labeled dataset with balancing and channel selection."""
    def __init__(self, path: str, transform=None, 
                 balance_dataset: Optional[Literal["upsample", "downsample", "augment_channels", "augment_time"]] = None, 
                 choose_channels: Optional[Iterable[int]] = None,
                 channel_names: Optional[Iterable[str]] = None):
        
        super().__init__(path, key="data")
        self.label_key = "labels"
        self.transform = transform
        self.choose_channels = choose_channels
        self.balance_dataset = balance_dataset

        if channel_names:
            self.channel_names = channel_names
        else:
            self.channel_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8","T7", "C3", "Cz", "C4", "T8","T5", "P3", "Pz", "P4", "T6","O1", "O2"]
            #self.channel_names = ['Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1','Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1','Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2','Fp2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2','FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9','FT9-FT10', 'FT10-T8', 'T8-P8-1'] # use for CHB-MIT dataset


        # Load metadata into memory for fast splitting/indexing
        self._sessions_all = torch.concat([self._get_tensor(p, "file_idxs") for p in self.paths])
        self._subjects_all = torch.concat([self._get_tensor(p, "subjects") for p in self.paths])
        self._labels_all = torch.cat([self._get_tensor(p, self.label_key) for p in self.paths])

        self.update_index_map(torch.arange(len(self._labels_all), dtype=torch.long))

    def _get_tensor(self, path, key):
        with h5py.File(path, "r") as file:
            return torch.from_numpy(file[key][:])

    def update_index_map(self, indices: torch.Tensor):
        """Filters the dataset to a specific subset of global indices."""
        cumulative = torch.tensor(self.cumulative_lengths, dtype=torch.long)
        file_indices = torch.bucketize(indices, cumulative[1:], right=True)
        local_indices = indices - cumulative[file_indices]
        
        self.index_map = {
            "indices": indices,
            "file_indices": file_indices,
            "local_indices": local_indices,
            "labels": self._labels_all[indices],
            "subjects": self._subjects_all[indices],
        }
        self.subjects = self.index_map["subjects"]
        self.labels = self.index_map["labels"]

    def apply_balancing(self):
        """Applies the balancing strategy defined in self.balance_dataset."""
        if not self.balance_dataset:
            return
        if self.balance_dataset in ["upsample", "augment_channels", "augment_time"]:
            self.upsample_minority_class()
        elif self.balance_dataset == "downsample":
            self.downsample_majority_class()

    def upsample_minority_class(self):
        labels = self.index_map["labels"]
        indices = self.index_map["indices"]
        distinct_labels, counts = torch.unique(labels, return_counts=True)
        max_count = counts.max().item()

        extra_indices = []
        for label, count in zip(distinct_labels, counts):
            deficit = max_count - int(count.item())
            if deficit > 0:
                label_pos = torch.nonzero(labels == label, as_tuple=True)[0]
                sampled_pos = label_pos[torch.randint(0, label_pos.numel(), (deficit,))]
                extra_indices.append(indices[sampled_pos])
        
        if extra_indices:
            self.update_index_map(torch.cat([indices] + extra_indices))

    def downsample_majority_class(self):
        labels = self.index_map["labels"]
        indices = self.index_map["indices"]
        distinct_labels, counts = torch.unique(labels, return_counts=True)
        min_count = counts.min().item()

        new_indices = []
        for label in distinct_labels:
            label_pos = torch.nonzero(labels == label, as_tuple=True)[0]
            sampled_pos = label_pos[torch.randperm(label_pos.numel())[:min_count]]
            new_indices.append(indices[sampled_pos])
        
        self.update_index_map(torch.cat(new_indices))

    def augment_time(self, data, drop_fraction=0.1):
        length = data.shape[-1]
        kept = length - int(length * drop_fraction)
        start = random.randint(0, length - kept)
        return data[:, start : start + kept]

    def augment_channels(self, data, drop_fraction=0.2):
        ch = data.shape[0]
        keep_n = ch - int(ch * drop_fraction)
        idx = random.sample(range(ch), keep_n)
        return data[idx, :], [self.channel_names[i] for i in idx]
    
    def __len__(self):
        return self.index_map["indices"].numel()

    def __getitem__(self, idx):
        file_idx = int(self.index_map["file_indices"][idx].item())
        local_idx = int(self.index_map["local_indices"][idx].item())
        
        with h5py.File(self.paths[file_idx], "r") as f:
            data = torch.from_numpy(f[self.key][local_idx])
        label = self.index_map["labels"][idx]

        if self.transform:
            data, label = self.transform((data, label))

        channels = self.channel_names

        if self.balance_dataset == "augment_channels":
            data, channels = self.augment_channels(data)
        elif self.balance_dataset == "augment_time":
            data = self.augment_time(data)

        #print(f"Sampled data shape: {data.shape}, Label: {label}, Channels: {channels}")

        return data, label, channels


def get_dataloaders(
    train_path: str,
    test_path: Optional[str] = None,
    transformer = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    n_splits: int = 1,
    train_balance_strategy: Optional[str] = None,
    split_info_path: str = "splits.json"
) -> Generator:
    """Generates fold-based dataloaders and saves subject split metadata."""
    base_ds = H5PYDatasetLabeled(train_path, transform=transformer)
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=seed)
    
    split_metadata = {}

    for fold, (train_val_idx, fold_test_idx) in enumerate(splitter.split(base_ds, groups=base_ds.subjects)):
        inner_split = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        t_idx, v_idx = next(inner_split.split(train_val_idx, groups=base_ds.subjects[train_val_idx]))
        
        # Convert to global indices
        actual_train_idx = torch.from_numpy(train_val_idx[t_idx])
        actual_val_idx = torch.from_numpy(train_val_idx[v_idx])
        actual_test_idx = torch.from_numpy(fold_test_idx)

        # Log metadata
        split_metadata[f"fold_{fold+1}"] = {
            "train_subjects": sorted(list(set(base_ds.subjects[actual_train_idx].tolist()))),
            "val_subjects": sorted(list(set(base_ds.subjects[actual_val_idx].tolist()))),
            "test_subjects": "external" if test_path else sorted(list(set(base_ds.subjects[actual_test_idx].tolist())))
        }

        # Create Loaders
        train_ds = H5PYDatasetLabeled(train_path, transform=transformer, balance_dataset=train_balance_strategy)
        train_ds.update_index_map(actual_train_idx)
        train_ds.apply_balancing()

        val_ds = H5PYDatasetLabeled(train_path, transform=transformer)
        val_ds.update_index_map(actual_val_idx)

        if test_path:
            test_ds = H5PYDatasetLabeled(test_path, transform=transformer)
        else:
            test_ds = H5PYDatasetLabeled(train_path, transform=transformer)
            test_ds.update_index_map(actual_test_idx)

        yield (
            fold,
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )

    with open(split_info_path, "w") as f:
        json.dump(split_metadata, f, indent=4)