import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class RadarDataset(Dataset):
   
    def __init__(self, hdf5_path, indices):
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.h5_file = None
        self.X = None
        self.Y = None
        self.Z = None 

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
            self.X = self.h5_file['X']
            self.Y = self.h5_file['Y']
            self.Z = self.h5_file['Z'] 

        data_idx = self.indices[idx]
        
        signal = self.X[data_idx].T
        label = np.argmax(self.Y[data_idx])
        snr = self.Z[data_idx][0] 
        
        signal_tensor = torch.from_numpy(signal).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        
        return signal_tensor, label_tensor, snr
