import traceback
import math
import torch
import numpy as np
import pytorch3d.ops
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph, KDTree

from models.utils import farthest_point_sampling
from .transforms import NormalizeUnitSphere


def patch_based_denoise(model, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1):
    """
    Args:
        pcl_noisy:  Input point cloud, (N, 3)
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
    N, d = pcl_noisy.size()
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
    patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]    # (N, K, 3)

    # Patch stitching preliminaries
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
    patches = patches - seed_pnts_1
    patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]
    patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)

    all_dists = torch.ones(num_patches, N) / 0
    all_dists = all_dists.cuda()
    all_dists = list(all_dists)
    patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(point_idxs_in_main_pcd)
    
    for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd, patch_dists): 
        all_dist[patch_id] = patch_dist

    all_dists = torch.stack(all_dists,dim=0)
    weights = torch.exp(-1 * all_dists)

    best_weights, best_weights_idx = torch.max(weights, dim=0)
    patches_denoised = []

    # Denoising
    i = 0
    patch_step = int(N / (seed_k_alpha * patch_size))
    assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
    while i < num_patches:
        # print("Processed {:d}/{:d} patches.".format(i, num_patches))
        curr_patches = patches[i:i+patch_step]
        try:
            patches_denoised_temp, _ = model.denoise_langevin_dynamics(curr_patches)
        except Exception as e:
            print("="*100)
            print(e)
            message_N = 5
            tb = ''.join(traceback.format_exception(e)[-message_N:])
            print(tb)
            print("="*100)
            print("If this is an Out Of Memory error, Seed_k_alpha might need to be increased to decrease patch_step.") 
            print("Additionally, if using multiple args.niters and a PyTorch3D ops, KNN, error arises, Seed_k might need to be increased to sample more patches for inference!")
            print("="*100)
            return
        patches_denoised.append(patches_denoised_temp)
        i += patch_step

    patches_denoised = torch.cat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_pnts_1
    
    # Patch stitching
    pcl_denoised = [patches_denoised[patch][point_idxs_in_main_pcd[patch] == pidx_in_main_pcd] for pidx_in_main_pcd, patch in enumerate(best_weights_idx)]

    pcl_denoised = torch.cat(pcl_denoised, dim=0)

    return pcl_denoised


def denoise_large_pointcloud(model, pcl, cluster_size, seed=0):
    device = pcl.device
    pcl = pcl.cpu().numpy()

    print('Running KMeans to construct clusters...')
    n_clusters = math.ceil(pcl.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(pcl)

    pcl_parts = []
    for i in tqdm(range(n_clusters), desc='Denoise Clusters'):
        pts_idx = kmeans.labels_ == i

        pcl_part_noisy = torch.FloatTensor(pcl[pts_idx]).to(device)
        pcl_part_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_part_noisy)
        pcl_part_denoised = patch_based_denoise(
            model,
            pcl_part_noisy,
            seed_k=5
        )
        pcl_part_denoised = pcl_part_denoised * scale + center
        pcl_parts.append(pcl_part_denoised)

    return torch.cat(pcl_parts, dim=0)
