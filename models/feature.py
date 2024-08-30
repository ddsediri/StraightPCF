import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList
import pytorch3d.ops
from .utils import *
from models.dynamic_edge_conv import DynamicEdgeConv
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.inits import reset

def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
    _, knn_idx, _ = pytorch3d.ops.knn_points(y, x, K=k+offset)
    return knn_idx[:, :, offset:]

class FeatureExtraction(Module):
    def __init__(self, k=32, input_dim=0, embedding_dim=512, distance_estimation=False):
        super(FeatureExtraction, self).__init__()
        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.distance_estimation = distance_estimation

        self.conv1 = DynamicEdgeConv(self.input_dim, int(self.embedding_dim / 8))
        self.conv2 = DynamicEdgeConv(int(self.embedding_dim / 8), int(self.embedding_dim / 4))
        self.conv3 = DynamicEdgeConv(int(self.embedding_dim / 8) + int(self.embedding_dim / 4), self.embedding_dim, activation=None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.conv1)
        reset(self.conv2)
        reset(self.conv3)

    def get_edge_index(self, x):
        cols = get_knn_idx(x, x, self.k+1).reshape(self.batch_size, self.num_points, -1)
        cols = (cols + self.rows_add).reshape(1, -1)
        edge_index = torch.cat([cols, self.rows], dim=0)
        edge_index, _ = remove_self_loops(edge_index.long())

        return edge_index
    
    def normalize_patch(self, pcl):
        scale = (pcl ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0]
        return pcl / scale

    def forward(self, x):
        self.batch_size = x.size(0)
        self.num_points = x.size(1)

        if self.distance_estimation:
            x = self.normalize_patch(x)

        self.rows = torch.arange(0, self.num_points).unsqueeze(0).unsqueeze(2).repeat(self.batch_size, 1, self.k+1).cuda()
        self.rows_add = (self.num_points*torch.arange(0, self.batch_size)).unsqueeze(1).unsqueeze(2).repeat(1, self.num_points, self.k+1).cuda()
        self.rows = (self.rows + self.rows_add).reshape(1, -1)        
        
        edge_index = self.get_edge_index(x)
        x = x.reshape(self.batch_size*self.num_points, -1)
        x1 = self.conv1(x, edge_index)
        x1 = x1.reshape(self.batch_size, self.num_points, -1)

        edge_index = self.get_edge_index(x1)
        x1 = x1.reshape(self.batch_size*self.num_points, -1)
        x2 = self.conv2(x1, edge_index)
        x2 = x2.reshape(self.batch_size, self.num_points, -1)

        edge_index = self.get_edge_index(x2)
        x2 = x2.reshape(self.batch_size*self.num_points, -1)
        x_combined = torch.cat((x1, x2), dim=-1)
        x_combined = x_combined.reshape(self.batch_size*self.num_points, -1)
        x = self.conv3(x_combined, edge_index)
        x = x.reshape(self.batch_size, self.num_points, -1)

        x, x_combined = x.reshape(self.batch_size, self.num_points, -1), x_combined.reshape(self.batch_size, self.num_points, -1)

        return x