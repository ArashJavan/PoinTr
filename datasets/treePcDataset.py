import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import logging

@DATASETS.register_module()
class TreePCDataset(data.Dataset):
    def __init__(self, config):
        self.subset = config.subset
        self.data_root = config.DATA_PATH % self.subset  
        self.partial_points_path = os.path.join(self.data_root, "partial")
        self.complete_points_path = os.path.join(self.data_root, "gt")  
        self.npoints = config.N_POINTS
        self.file_list = self._get_file_list()

        print(f'[DATASET] will use {len(self.file_list)} instances')
        
    def _get_file_list(self):
        fname = os.path.join(self.data_root, "indices.txt")
        file_list = None 
        with open(fname, "r") as f:
            file_list = f.read().splitlines()
        return file_list
        
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def __getitem__(self, idx):
        fname = self.file_list[idx]
        fpath = os.path.join(self.complete_points_path, fname)
        gt_data = IO.get(fpath).astype(np.float32)
        gt_data = torch.from_numpy(gt_data).float()
        
        fpath = os.path.join(self.partial_points_path, fname)
        partial_data = IO.get(fpath).astype(np.float32)
        partial_data = torch.from_numpy(partial_data).float()

        model_id = "tree_model"
        taxonomy_id = "tree"
        return model_id, taxonomy_id, (partial_data, gt_data)

    def __len__(self):
        return len(self.file_list)