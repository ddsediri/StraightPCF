import random
import torch
from torch.utils.data import Dataset
from pytorch3d.ops import knn_points
from tqdm.auto import tqdm


def make_patches_for_pcl_pair(pcl_A, pcl_A_L1, pcl_A_L2, pcl_B, patch_size, num_patches, ratio, train_cvm_network, tot_its=None):
    """
    Args:
        pcl_A:  The first point cloud, (N, 3).
        pcl_B:  The second point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """

    if not train_cvm_network:
        N = pcl_A_L2.size(0)
        seed_idx = torch.randperm(N)[:num_patches]   # (P, ) P=1
        seed_pnts = pcl_A_L2[seed_idx].unsqueeze(0)   # (1, P, 3) P=1

        _, nn_idx, pat_A = knn_points(seed_pnts, pcl_A_L2.unsqueeze(0), K=patch_size, return_nn=True)
        pat_A = pat_A[0]  # (1, M, 3)

        pat_B = pcl_B[nn_idx.squeeze(), :].unsqueeze(0)

        l1 = 1e-8
        l2 = 1.0

        t = (l2 -l1) * torch.rand(1, 1, 1).repeat(1, patch_size, 1) + l1
        pat_t = t * pat_B + (1 - t) * pat_A
        seed_points_t = t * pcl_B[seed_idx].unsqueeze(0) + (1 - t) * pcl_A_L2[seed_idx].unsqueeze(0)
         
        pat_A = pat_A - seed_points_t
        pat_B = pat_B - seed_points_t
        pat_t = pat_t - seed_points_t

        return pat_A, pat_B, pat_t
    elif train_cvm_network:
        N = pcl_A_L2.size(0)
        seed_idx = torch.randperm(N)[:num_patches]   # (P, ) P=1
        seed_pnts = pcl_A_L2[seed_idx].unsqueeze(0)   # (1, P, 3) P=1

        _, nn_idx, pat_A = knn_points(seed_pnts, pcl_A_L2.unsqueeze(0), K=patch_size, return_nn=True)
        pat_A = pat_A[0]  # (1, M, 3)

        pat_B = pcl_B[nn_idx.squeeze(), :].unsqueeze(0)

        l1 = 1e-8
        l2 = 1.0

        t = (l2 -l1) * torch.rand(1) + l1
        original_time_step = t
        
        t = t.unsqueeze(1).unsqueeze(2).repeat(1, patch_size, 1)

        seed_points_t = t * pcl_B[seed_idx].unsqueeze(0) + (1 - t) * pcl_A_L2[seed_idx].unsqueeze(0)

        return pat_A, pat_B, seed_points_t, original_time_step
    else:
        raise Exception("Invalid setting for patch generation!")
    
        return
    

class PairedPatchDataset(Dataset):

    def __init__(self, datasets, patch_ratio, train_cvm_network=False, tot_its=10, patch_size=1000, num_patches=1000, transform=None):
        super().__init__()
        self.datasets = datasets
        self.len_datasets = sum([len(dset) for dset in datasets])
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.train_cvm_network = train_cvm_network
        self.transform = transform
        self.tot_its = tot_its

    def __len__(self):
        return self.len_datasets * self.num_patches


    def __getitem__(self, idx):
        pcl_dset = random.choice(self.datasets)
        pcl_data = pcl_dset[idx % len(pcl_dset)]
        pat_std = torch.tensor(pcl_data['noise_std'], dtype=torch.float)
        
        if not self.train_cvm_network:
            pat_noisy_L2, pat_clean, pat_noisy = make_patches_for_pcl_pair(
                pcl_data['pcl_noisy'],
                pcl_data['pcl_noisy_L1'],
                pcl_data['pcl_noisy_L2'],
                pcl_data['pcl_clean'],
                patch_size=self.patch_size,
                num_patches=1,
                ratio=self.patch_ratio,
                train_cvm_network=self.train_cvm_network
            )
            data = {
                'pcl_noisy_L2': pat_noisy_L2[0],
                'pcl_clean': pat_clean[0],
                'pcl_noisy': pat_noisy[0],
                'pcl_std'  : pat_std,
            }
        else:
            pat_noisy_L2, pat_clean, seed_points_t, original_time_step = make_patches_for_pcl_pair(
                pcl_data['pcl_noisy'],
                pcl_data['pcl_noisy_L1'],
                pcl_data['pcl_noisy_L2'],
                pcl_data['pcl_clean'],
                patch_size=self.patch_size,
                num_patches=1,
                ratio=self.patch_ratio,
                train_cvm_network=self.train_cvm_network,
                tot_its=self.tot_its
            )
            data = {
                'pcl_noisy_L2': pat_noisy_L2[0],
                'pcl_clean': pat_clean[0],
                'seed_points_t': seed_points_t[0],
                'original_time_step': original_time_step[0],
                'pcl_std'  : pat_std,
            }


        if self.transform is not None:
            data = self.transform(data)
        return data
