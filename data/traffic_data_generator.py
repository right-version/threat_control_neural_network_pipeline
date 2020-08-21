import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class TrafficDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file: path to the .csv file with preprocessed features
        """
        self.csf_file = csv_file
        self.data = pd.read_csv(self.csf_file)

        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data[idx])
        tensor.to(torch.float)
        return tensor

