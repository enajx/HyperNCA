import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)


def make_sequental_mlp(input_space, action_space, hidden_dim, bias, layers): 
    
    fc_in = nn.Linear(input_space, hidden_dim, bias=bias)
    fc_out = nn.Linear(hidden_dim, action_space , bias=bias)
    tanh = torch.nn.Tanh()
    layer_list = [fc_in, tanh]
    for i in range(1, layers-1):
        layer_list.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        layer_list.append(torch.nn.Tanh())
    layer_list.append(fc_out)
    
    return torch.nn.Sequential(*layer_list)


class MLPn(nn.Module):        
    
    def __init__(self, input_space, action_space, hidden_dim, bias, layers):
        super(MLPn, self).__init__()
        self.out = make_sequental_mlp(input_space, action_space, hidden_dim, bias, layers)

    def forward(self, x):
        return self.out(x)