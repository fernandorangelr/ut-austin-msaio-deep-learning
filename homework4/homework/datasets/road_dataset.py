from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import road_transforms
from .road_utils import Track


class RoadDataset(Dataset):
    """
    SuperTux dataset for road detection
    """

    def __init__(
        self,
        episode_path: str,
        transform_pipeline: str = "default",
    ):
        super().__init__()

        self.episode_path = Path(episode_path)

        info = np.load(self.episode_path / "info.npz", allow_pickle=True)

        self.track = Track(**info["track"].item())
        self.frames: dict[str, np.ndarray] = {k: np.stack(v) for k, v in info["frames"].item().items()}
        self.transform = self.get_transform(transform_pipeline)

    def get_transform(self, transform_pipeline: str):
        """
        Creates a pipeline for processing data.

        Feel free to add your own pipelines (e.g. for data augmentation).
        Note that the grader will choose one of the predefined pipelines,
        so be careful if you modify the existing ones.
        """
        xform = None

        if transform_pipeline == "default":
            # image, track_left, track_right, waypoints, waypoints_mask
            xform = road_transforms.Compose(
                [
                    road_transforms.ImageLoader(self.episode_path),
                    road_transforms.EgoTrackProcessor(self.track),
                ]
            )
        elif transform_pipeline == "state_only":
            # track_left, track_right, waypoints, waypoints_mask
            xform = road_transforms.EgoTrackProcessor(self.track)
        elif transform_pipeline == "aug":
            # add your custom augmentations here
            pass

        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")

        return xform

    def __len__(self):
        return len(self.frames["location"])

    def __getitem__(self, idx: int):
        sample = {"_idx": idx, "_frames": self.frames}
        sample = self.transform(sample)

        # remove private keys
        for key in list(sample.keys()):
            if key.startswith("_"):
                sample.pop(key)

        return sample


def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 2,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader | Dataset:
    """
    Constructs the dataset/dataloader.
    The specified transform_pipeline must be implemented in the RoadDataset class.

    Args:
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        return_dataloader (bool): returns either DataLoader or Dataset
        num_workers (int): data workers, set to 0 for VSCode debugging
        batch_size (int): batch size
        shuffle (bool): should be true for train and false for val

    Returns:
        DataLoader or Dataset
    """
    dataset_path = Path(dataset_path)
    scenes = [x for x in dataset_path.iterdir() if x.is_dir()]

    # can pass in a single scene like "road_data/val/cornfield_crossing_04"
    if not scenes and dataset_path.is_dir():
        scenes = [dataset_path]

    datasets = []
    for episode_path in sorted(scenes):
        datasets.append(RoadDataset(episode_path, transform_pipeline=transform_pipeline))
    dataset = ConcatDataset(datasets)

    print(f"Loaded {len(dataset)} samples from {len(datasets)} episodes")

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
    )
