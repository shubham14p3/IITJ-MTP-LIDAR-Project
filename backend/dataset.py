import os
import glob
import numpy as np
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    """
    Dataset for processed .npz files with:
      - points: (N,7) float32 [x,y,z,r,g,b,intensity]
      - labels: (N,) int64     # dummy / real labels
    """

    def __init__(self, root_dir, split='train', num_points=2048):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        pattern = os.path.join(root_dir, f"{split}_*.npz")
        self.files = sorted(glob.glob(pattern))

        if not self.files:
            raise RuntimeError(
                f"No files found for split={split} in {root_dir}. "
                f"Did you run preprocess_3dses.py ?"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)

        points = data["points"]  # (N,7)
        labels = data["labels"]  # (N,)

        N = points.shape[0]

        # Random sampling to self.num_points
        if N >= self.num_points:
            idxs = np.random.choice(N, self.num_points, replace=False)
        else:
            pad = np.random.choice(N, self.num_points - N, replace=True)
            idxs = np.concatenate([np.arange(N), pad])

        # Use ALL 7 features: xyz + rgb + intensity
        pts = points[idxs, :7]    # <--------- FIXED HERE

        lbl = labels[idxs]

        return pts.astype("float32"), lbl.astype("int64")
