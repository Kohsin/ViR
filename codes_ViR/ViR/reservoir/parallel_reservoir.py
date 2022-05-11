import torch
from torch import nn
from einops.layers.torch import Rearrange

from reservoir.cycle_reservoir_layer import ReservoirLayer

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Reservoir(nn.Module):
    def __init__(self, dim, depth, mlp_dim, device, dropout = 0., input_scaling=1, spectral_radius=0.9, leaky=1,
                 sparsity=0.05, reservoir_units = 1000, cycle_weight=0.05, jump_weight=0.5, jump_size=137, connection_weight=0.08):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.device = device
        self.depth = depth
        for i in range(self.depth):
            self.layers.append(nn.ModuleList([
                ReservoirLayer(
                    input_size=dim,
                    units=reservoir_units,
                    input_scaling=input_scaling,
                    spectral_radius=spectral_radius,
                    leaky=leaky,
                    sparsity=sparsity,
                    output_size=dim,
                    cycle_weight=cycle_weight,
                    jump_weight=jump_weight,
                    jump_size=jump_size,
                    connection_weight=connection_weight),
                ReservoirLayer(
                    input_size=dim,
                    units=reservoir_units,
                    input_scaling=input_scaling,
                    spectral_radius=spectral_radius,
                    leaky=leaky,
                    sparsity=sparsity,
                    output_size=dim,
                    cycle_weight=cycle_weight,
                    jump_weight=jump_weight,
                    jump_size=jump_size,
                    connection_weight=connection_weight),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x1, x2, mask = None):
        total_output = torch.zeros(torch.cat((x1,x2)).shape).to(self.device)

        for i, layer in enumerate(self.layers):
            #reservoir1, ff = layer
            reservoir1, reservoir2, ff = layer
            #output = reservoir(x)
            
            output1 = reservoir1(x1)
            output2 = reservoir2(x2)
            output = torch.cat((output1,output2))
            output = output.view(100,-1)
            output = ff(output)
            total_output += output
        return total_output / self.depth

class Parallel_Reservoir(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, mlp_dim, device, channels=3, dropout=0.,
                 input_scaling=1, spectral_radius=0.9, leaky=1, sparsity=0.05, reservoir_units=1000, cycle_weight=0.05, jump_weight=0.5, jump_size=137, connection_weight=0.08):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        self.dropout = nn.Dropout(dropout)

        self.reservoir = Reservoir(
                            dim=dim,
                            depth=depth,
                            mlp_dim=mlp_dim,
                            device=device,
                            dropout=dropout,
                            input_scaling=input_scaling,
                            spectral_radius=spectral_radius,
                            leaky=leaky,
                            sparsity=sparsity,
                            reservoir_units=reservoir_units,
                            cycle_weight=cycle_weight,
                            jump_weight=jump_weight,
                            jump_size=jump_size,
                            connection_weight=connection_weight)

        self.to_latent = nn.Identity()

        input_dim = dim * (round(image_size / patch_size) ** 2)

        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes)
        )
        

    def forward(self, img, mask = None):
        print("\n")
        print("image.shape",img.shape)
        #x = self.to_patch_embedding(img)
        #b, n, _ = x.shape
        #print("x.shape1",x.shape)
        #x = self.dropout(x)
        
        x1, _, x2 = torch.svd(img)
        x1 = self.to_patch_embedding(x1)
        x2 = self.to_patch_embedding(x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x = self.reservoir(x1, x2, mask)
        
        #print("x.shape2",x.shape)
        
        #x = self.to_patch_embedding(img)
        #x = self.dropout(x)
        #x = self.reservoir(x)
        print("x.shape3",x.shape)
        x = x.view(x.shape[0], -1)
        print("x.shape4",x.shape)
        x = self.to_latent(x)
        print("x.shape5",x.shape)
        x = self.mlp_head(x)
        print("x.shape6",x.shape)

        return x
