import torch
from torch import nn
from torch.distributions.uniform import Uniform
from pytorch3d.ops import knn_points
from pytorch3d.loss.chamfer import chamfer_distance
import numpy as np

from models.vm import VelocityModule

from inspect import signature

def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]

class CoupledVMArch(nn.Module):

    def __init__(self, velocity_nets=None, args=None):
        super().__init__()
        self.args = args
        # geometry
        self.frame_knn = args.frame_knn
        self.tot_its = 3
        self.num_train_points = args.num_train_points
        # score-matching
        self.dsm_sigma = args.dsm_sigma
        # networks

        if hasattr(args, "num_modules"):
            self.num_modules = args.num_modules
        else:
            self.num_modules = 2
            args.velocity_ckpt = './pretrained_vm/ckpt_vm.pt'

        if velocity_nets is not None:
            self.velocity_nets = velocity_nets
            self.num_modules = len(self.velocity_nets)
        else:
            velocity_ckpt = torch.load(args.velocity_ckpt, map_location=args.device)
            self.velocity_nets = nn.ModuleList()
            for i in range(self.num_modules):
                self.velocity_nets.append(VelocityModule(args=velocity_ckpt['args']).to(args.device))

    def get_supervised_loss(self, pcl_clean, pcl_noisy_L2, pcl_seeds_t, original_time_step):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
        """
        B, N, d = pcl_noisy_L2.size()
        pnt_idx = get_random_indices(N, self.num_train_points)

        grad_target = (pcl_clean - pcl_noisy_L2)

        total_dir_loss = 0.0
        total_consistency_loss = 0.0

        curr_step = (original_time_step * (self.num_modules - 0) + 0) / self.num_modules
        curr_step = curr_step.unsqueeze(1).unsqueeze(2)
        pcl_noisy = curr_step * pcl_clean + (1 - curr_step) * pcl_noisy_L2
        pcl_noisy = pcl_noisy - pcl_seeds_t

        for mod in range(self.num_modules):
            feat = self.velocity_nets[mod].encoder(pcl_noisy)
            F = feat.size(2)
            pred_dir = self.velocity_nets[mod].decoder(c = feat.view(-1, F)).reshape(B, N, d) 
            dir_loss = (((pred_dir - grad_target) ** 2)).sum(dim=-1).mean()
            total_dir_loss += dir_loss

            pcl_noisy = pcl_noisy + ((1. - original_time_step.unsqueeze(1).unsqueeze(2)) / self.num_modules) * pred_dir

            if mod < self.num_modules - 1:
                curr_step_plus_1 = (original_time_step * (self.num_modules - (mod + 1)) + (mod + 1)) / self.num_modules
                curr_step_plus_1 = curr_step_plus_1.unsqueeze(1).unsqueeze(2)
                pcl_noisy_interpolated = curr_step_plus_1 * pcl_clean + (1 - curr_step_plus_1) * pcl_noisy_L2
                pcl_noisy_interpolated = pcl_noisy_interpolated - pcl_seeds_t
                consistency_loss = (((pcl_noisy_interpolated - pcl_noisy) ** 2)).sum(dim=-1).mean()
                total_consistency_loss += consistency_loss

        return (total_dir_loss + 10 * total_consistency_loss)/ self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()
        with torch.no_grad():
            pcl_next = pcl_noisy.clone()

            for it in range(self.tot_its):
                # Trajectories
                self.velocity_nets.eval()
                
                for mod in range(self.num_modules):
                    feat = self.velocity_nets[mod].encoder(pcl_next)
                    F = feat.size(2)

                    pred_dir = self.velocity_nets[mod].decoder(c = feat.view(-1, F)).reshape(B, N, d) 

                    pcl_next = pcl_next + (1 / self.tot_its) * (1 / self.num_modules) * pred_dir # 0.75
                        
        return pcl_next, None