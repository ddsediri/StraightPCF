import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, z_dim, dim, out_dim, hidden_size):
        """
        Args:
            z_dim:   Dimension of context vectors. 
            dim:     Point dimension.
            out_dim: Gradient dim.
            hidden_size:   Hidden states dim.
        """
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size

        # Input = Conditional = zdim (code) + dim (xyz)
        c_dim = z_dim #+ dim
        self.lin_1 = nn.Linear(c_dim, c_dim)
        self.bn_1_out = nn.BatchNorm1d(c_dim)
        self.lin_2 = nn.Linear(c_dim, hidden_size)        
        self.bn_2_out = nn.BatchNorm1d(hidden_size)
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        self.actvn_out = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    # def forward(self, x, c):
    def forward(self, c, B=None, N=None):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim) Shape latent code
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        net = self.dropout(self.actvn_out(self.bn_1_out(self.lin_1(c))))
        net = self.dropout(self.actvn_out(self.bn_2_out(self.lin_2(net))))

        if self.out_dim == 1:
            net = net.reshape(B, N, -1)
            net = net.max(dim=1, keepdim=True)[0]
            net = torch.sigmoid(self.lin_3(net))
        else:
            net = self.lin_3(net)
        return net