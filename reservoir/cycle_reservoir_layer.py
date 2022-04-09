import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
import random
import math
"""
def init_reservoir(units, cycle_weight, jump_weight, jump_size, connection_weight):
    sparse_matrix = torch.ones(units, units) * connection_weight
    random_sign = torch.rand(units, units)
    sparse_matrix[0][units - 1] = cycle_weight
    for i in range(units - 1):
        sparse_matrix[i + 1][i] = cycle_weight
    jumped_unit = 0
    while (jumped_unit + jump_size) <= units:
        if (jumped_unit + jump_size) == units:
            sparse_matrix[jumped_unit][0] = jump_weight
            sparse_matrix[0][jumped_unit] = jump_weight
        else:
            sparse_matrix[jumped_unit][jumped_unit + jump_size] = jump_weight
            sparse_matrix[jumped_unit + jump_size][jumped_unit] = jump_weight
        jumped_unit += jump_size

    for i in range(units):
        for j in range(units):
            sign = 1
            if random_sign[i][j] < 0.5:
                sign = -1
            sparse_matrix[i][j] *= sign
    c = int(random.randint(0, units))
    m = int(random.randint(0, units))
    sparse_matrix[c][m] = 0
    return sparse_matrix
""""
def init_weight(dim_x, dim_y, sparsity):
    sparse_matrix = sp.rand(dim_x, dim_y, density=sparsity, format='csr').toarray()
    return sparse_matrix

def spectral_norm_scaling(W: torch.Tensor, rho_desired: float) -> torch.Tensor:
    e, _ = np.linalg.eig(W)
    rho_curr = max(abs(e))
    return W * (rho_desired / rho_curr)



class Cycle_Reservoir(torch.nn.Module):
    def __init__(self, input_size, units, input_scaling=1., spectral_radius=0.9, leaky=1, output_size=128,
                 sparsity=0.05, cycle_weight=0.05, jump_weight=0.5, jump_size=137, connection_weight=0.08):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.sparsity = sparsity
        self.kernel = torch.Tensor(init_weight(input_size, self.units, sparsity * self.input_scaling)) #W_in
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)
        
        self.softplus = nn.Identity()
        self._epsilon = 1e-8
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._allocate_parameters()
        """
        W = init_reservoir(self.units, cycle_weight, jump_weight, jump_size, connection_weight)
        if self.leaky == 1:
            W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = W
        else:
            I = torch.ones(self.units, self.units)
            W = torch.Tensor(W * self.leaky + (I * (1 - self.leaky)))
            W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = (W + I * (self.leaky - 1)) * (1 / self.leaky) #W
        self.recurrent_kernel = nn.Parameter(self.recurrent_kernel, requires_grad=False)
"""
        self.weight_out = nn.Linear(self.units + self.output_size, self.output_size)
        self.weight = nn.Linear(self.units * 2 + self.output_size * 4, self.output_size)

        self.bias = torch.Tensor((torch.rand(self.units + self.input_size) * 2 - 1) * self.input_scaling)
        self.bias = nn.Parameter(self.bias, requires_grad=False)



    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value,requires_grad=False)
        self.register_parameter(name, param)
        return param
    
    def ode_solver(self, state, inputs):
        state_pre = state
        #print("_sigmoid().shape", self._sigmoid(
        #  inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        #).shape)
        sensory_w_activation = self.softplus(self._params["sensory_w"]) * self._sigmoid(
           inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        cm_t = self.softplus(self._params["cm"]) / (
            (0.0001 + 1) / 6
        )

        w_numerator_sensory = torch.sum(sensory_rev_activation)
        w_denominator_sensory = torch.sum(sensory_w_activation)

        for t in range(6):
           w_activation = self.softplus(self._params["w"]) * self._sigmoid(
               state_pre, self._params["mu"], self._params["sigma"]
           )

           rev_activation = w_activation * self._params["erev"]

           # Reduce over dimension 1 (=source neurons)
           w_numerator = torch.sum(rev_activation) + w_numerator_sensory
           w_denominator = torch.sum(w_activation) + w_denominator_sensory

           numerator = (
               cm_t * state_pre
               + self.softplus(self._params["gleak"]) * self._params["vleak"]
               + w_numerator
           )
           denominator = cm_t + self.softplus(self._params["gleak"]) + w_denominator
           
           # Avoid dividing by 0
           state_pre = numerator / (denominator + self._epsilon)
        return state_pre
    
    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((100,1000), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((100,1000), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((100,1000), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (1000, 1000), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((1000, 1000), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((100, 1000), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(
                self._erev_initializer((100,1000))
            ),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (1000, 1000), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (1000, 1000), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (100,1000), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(
                self._erev_initializer((100,1000))
            ),
        )
    
    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval
    
    def _erev_initializer(self, shape=None):
        return np.random.default_rng().choice([-1, 1], size=shape)
    
    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        x = torch.sum(x, dim=2)
        return torch.tanh(x)
    
    def forward(self, xt, h_prev):
        x = xt.clone().detach()
        #ts = ts.view((-1, 1))
        #self.recurrent_kernel.datach_()
        #self.kernel.dtach_()
        input_part = torch.mm(x, self.kernel)
        
        state_part = self.ode_solver(h_prev, input_part)
        #print("test", test.size())
        #state_part = torch.tanh(torch.mm(h_prev , self.recurrent_kernel) + input_part)
        
        output = torch.cat([x, h_prev * (1 - self.leaky) + state_part], dim=1)
        reservoir_output = torch.tanh(self.weight_out(output))
        reservoir_output1 = torch.cat([x, state_part, reservoir_output,
                                      x ** 2, state_part ** 2,  reservoir_output ** 2], dim=1)
        reservoir_output2 = torch.tanh(self.weight(reservoir_output1))
        return state_part, reservoir_output2


class ReservoirLayer(torch.nn.Module):
    def __init__(self, input_size, units, input_scaling=1, spectral_radius=0.9, leaky=1, sparsity=0.05,
                 output_size=128, cycle_weight=0.05, jump_weight=0.5, jump_size=137, connection_weight=0.08):
        super().__init__()
        self.cycle_reservoir_layer = Cycle_Reservoir(input_size, units, input_scaling, spectral_radius, leaky,
                                                     output_size, sparsity, cycle_weight, jump_weight, jump_size, connection_weight)
        self.h_prev = None

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.cycle_reservoir_layer.units)

    def set_reservoir_state(self, state):
        self.h_prev = state

    def forward(self, x):
        if self.h_prev is None:
            self.h_prev = self.init_hidden(x.shape[0]).to(x.device)

        hs = []

        for t in range(x.shape[1]):
            xt = x[:, t]
            self.h_prev, output = self.cycle_reservoir_layer(xt, self.h_prev)
            hs.append(output)
        
        hs = torch.stack(hs, dim=1)

        return hs
