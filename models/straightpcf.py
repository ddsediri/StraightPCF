import torch
from torch import nn
from torch.distributions.uniform import Uniform
from pytorch3d.ops import knn_points
from pytorch3d.loss.chamfer import chamfer_distance
import numpy as np
from torch_geometric.utils import get_laplacian, to_dense_adj
from models.vm import VelocityModule
from models.feature import FeatureExtraction
from models.decoder import Decoder

from inspect import signature

def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]

class StraightPCF(nn.Module):

    def __init__(self, straightpcf=None, args=None):
        super().__init__()
        self.args = args
        # geometry
        self.frame_knn = args.frame_knn
        self.tot_its = args.tot_its
        self.num_train_points = args.num_train_points
        # score-matching
        self.dsm_sigma = args.dsm_sigma
        # networks
        if straightpcf is not None:
            self.velocity_nets = straightpcf.velocity_nets
            self.num_modules = len(self.velocity_nets)
        else:
            if not hasattr(args, 'cvm_ckpt'):
                args.cvm_ckpt = './pretrained_cvm/ckpt_cvm.pt'
            cvm_ckpt = torch.load(args.cvm_ckpt, map_location=args.device)

            if hasattr(cvm_ckpt['args'], 'num_modules'):
                self.num_modules = cvm_ckpt['args'].num_modules
            else:
                self.num_modules = 2

            self.velocity_nets = nn.ModuleList()
            for i in range(self.num_modules):
                self.velocity_nets.append(VelocityModule(args=cvm_ckpt['args']).to(args.device))
        
        self.encoder = FeatureExtraction(k=self.frame_knn, 
                                         input_dim=3, 
                                         embedding_dim=args.feat_embedding_dim, 
                                         distance_estimation=args.distance_estimation)
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3, 
            out_dim=1,
            hidden_size=args.decoder_hidden_dim,
        )

    def get_supervised_loss(self, pcl_clean, pcl_noisy_L2, pcl_seeds_t, original_time_step):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
        """
        B, N, d = pcl_noisy_L2.size()
        pnt_idx = get_random_indices(N, self.num_train_points)

        curr_step = original_time_step.unsqueeze(1).unsqueeze(2)
        pcl_noisy = curr_step * pcl_clean + (1 - curr_step) * pcl_noisy_L2

        num = torch.sqrt(((pcl_clean - pcl_noisy) ** 2).sum(dim=-1))
        den = torch.sqrt(((pcl_clean - pcl_noisy_L2) ** 2).sum(dim=-1))
        ratio = num[:, 0] / den[:, 0]

        pcl_clean = pcl_clean - pcl_seeds_t
        pcl_noisy = pcl_noisy - pcl_seeds_t

        feat_d = self.encoder(pcl_noisy)
        F_d = feat_d.size(2)
        pred_d = self.decoder(c = feat_d.view(-1, F_d), B = B, N = N).reshape(B) 

        loss = ((pred_d - ratio) ** 2).mean()

        for mod in range(self.num_modules):
            feat = self.velocity_nets[mod].encoder(pcl_noisy)
            F = feat.size(2)
            pred_dir = self.velocity_nets[mod].decoder(c = feat.view(-1, F)).reshape(B, N, d) 
            pcl_noisy = pcl_noisy + (1 / self.num_modules) * pred_d.view(B, 1, 1) * pred_dir

        finetune_loss = 2e2 * ((pcl_clean - pcl_noisy) ** 2).sum(dim=-1).mean()

        return (loss + finetune_loss) / self.dsm_sigma
    
    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()
        with torch.no_grad():
            pcl_next = pcl_noisy.clone()

            self.eval()

            feat_d = self.encoder(pcl_next)
            F_d = feat_d.size(2)
            pred_d = self.decoder(c = feat_d.view(-1, F_d), B = B, N = N).reshape(B, 1, 1) 

            for it in range(self.tot_its):
                # Trajectories
                
                pred_disp = torch.zeros(B, N, d)
                for mod in range(self.num_modules):
                    feat = self.velocity_nets[mod].encoder(pcl_next)
                    F = feat.size(2)

                    pred_dir = self.velocity_nets[mod].decoder(c = feat.view(-1, F)).reshape(B, N, d) 
                    pred_disp = (1 / self.tot_its) * (1 / self.num_modules) * pred_d * pred_dir
                    pcl_next = pcl_next + pred_disp

                    # mat = self.glr(pcl_next)
                    # pcl_next = torch.bmm(mat, pcl_next)
                
                        
        return pcl_next, None