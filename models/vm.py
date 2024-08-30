import torch
from torch import nn
from torch.distributions.uniform import Uniform
from pytorch3d.ops import knn_points
import numpy as np

from models.feature import FeatureExtraction
from models.decoder import Decoder

def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]


class VelocityModule(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        # geometry
        self.frame_knn = args.frame_knn
        self.num_train_points = args.num_train_points
        # score-matching
        self.dsm_sigma = args.dsm_sigma
        # networks
        self.encoder = FeatureExtraction(k=self.frame_knn, input_dim=3, embedding_dim=args.feat_embedding_dim)
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3, 
            out_dim=3,
            hidden_size=args.decoder_hidden_dim,
        )

    def get_supervised_loss(self, pcl_noisy_L2, pcl_noisy, pcl_clean, pcl_std):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
        """
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        feat = self.encoder(pcl_noisy)  # (B, N, F)
        F = feat.size(2)

        # Feature extraction
        feat = feat[:,pnt_idx,:]
        pcl_noisy_L2 = pcl_noisy_L2[:,pnt_idx,:] 
        pcl_noisy = pcl_noisy[:,pnt_idx,:]  
        pcl_clean = pcl_clean[:,pnt_idx,:]           
        
        grad_dir_t_target = pcl_clean - pcl_noisy_L2

        pred_dir = self.decoder(c = feat.view(-1, F)).reshape(B, len(pnt_idx), d) 

        loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        
        return loss
    
    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()
        tot_steps = 4
        with torch.no_grad():
            pcl_next = pcl_noisy.clone()

            for it in range(tot_steps):
                # Trajectories
                self.encoder.eval()
                self.decoder.eval()
                
                feat = self.encoder(pcl_next)
                F = feat.size(2)

                frame_centered = pcl_next.unsqueeze(2)
                pred_dir = self.decoder(c = feat.view(-1, F)).reshape(B, N, d) 
                pcl_next += (1 / tot_steps) * pred_dir
                        
        return pcl_next, None