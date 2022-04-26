import torch
import copy
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot
import time

from visualisation_utils import visualiseVoxs2Dmulti, visualiseNetwork


def merge_dicts_helper(dict1, dict2):
    """ Recursively merges dict2 into dict1 """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            dict1[k] = merge_dicts_helper(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1

def merge_dicts(dict1,dict2):
    dict1 = copy.deepcopy(dict1)
    dict2 = copy.deepcopy(dict2)
    return merge_dicts_helper(dict1, dict2)

class ConfigBase:
    DEFAULT_CONFIG = {}
    REQUIRED_CONFIG = {}

    def __init__(self, config):
        self.config = merge_dicts(self.__class__.DEFAULT_CONFIG, config)
        self.check_required_config()

    def check_required_config(self):
        for r in self.REQUIRED_CONFIG:
            if r not in self.config:
                raise ValueError("required key: {}, not found in config: {}".format(r, self.config))

class TorchModule(torch.nn.Module, ConfigBase):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        ConfigBase.__init__(self, config)

def make_sequental_3d(in_channels, out_channels, bias): 
    conv3d = torch.nn.Conv3d(in_channels*3, out_channels[0], kernel_size=1, bias=bias)
    tanh = torch.nn.Tanh()
    layer_list = [conv3d, tanh]
    for i in range(1, len(out_channels)):
        layer_list.append(torch.nn.Conv3d(out_channels[i-1], out_channels[i], kernel_size=1, bias=bias))
        layer_list.append(torch.nn.Tanh())
    layer_list.append(torch.nn.Conv3d(out_channels[-1], in_channels, kernel_size=1, bias=bias))
    # layer_list.append(torch.nn.Conv3d(out_channels[-1], in_channels, kernel_size=1, padding_mode='circular', bias=bias))
    return torch.nn.Sequential(*layer_list)

class SmallerCellUpdateNet(torch.nn.Module):        
    """
    Cells’ update rule: each cell applies a series of operations to the perception vector (same rule for all the cell)
    """
    def __init__(self, in_channels, out_channels, bias):
        super(SmallerCellUpdateNet, self).__init__()
        self.out = make_sequental_3d(in_channels, out_channels, bias)

    def forward(self, x):
        return self.out(x)
    
    
class CellPerceptionNet(torch.nn.Module):           
    """
    Defines what each cell perceives of the environment surrounding it
    In original Mordvintsev NCA, CNN parameters weren't learnt but Sobel filters were used, here we learned the CNN parameters.
    """
    def __init__(self, in_channels, bias):
        super(CellPerceptionNet, self).__init__()
        self.num_channels = in_channels            
        self.bias = bias            
        self.conv1 = torch.nn.Conv3d(in_channels=self.num_channels, out_channels=self.num_channels*3, kernel_size=3, stride=1, padding=1, groups=self.num_channels, bias=self.bias)
        # self.conv1 = torch.nn.Conv3d(in_channels=self.num_channels, out_channels=self.num_channels*3, kernel_size=3, stride=1, padding=1, padding_mode='circular', groups=self.num_channels, bias=self.bias)

    def forward(self, x):
        return self.conv1(x)


class CellCAModel3D(TorchModule):
    DEFAULT_CONFIG = {
        "perception_net_class":CellPerceptionNet,
        "update_net_class":SmallerCellUpdateNet,
    }

    def __init__(self, config):
        super(CellCAModel3D, self).__init__(config)
        self.device = self.config.get('device')
        self.num_channels =  self.config.get('NCA_channels')
        self.bias = self.config.get('NCA_bias') 
        self.update_net_channel_dims = [self.config.get('update_net_channel_dims'), self.config.get('update_net_channel_dims') ]
        self.living_channel_dim = self.config.get("living_channel_dim")
        self.num_categories = self.living_channel_dim
        self.alpha_living_threshold =  self.config.get('alpha_living_threshold') 
        self.perception_net_class = self.config.get("perception_net_class")
        self.update_net_class = self.config.get("update_net_class")
        self.num_categories = self.config.get('num_categories')
        self.perception_net = self.perception_net_class(self.num_channels, self.bias)
        for p in self.perception_net.parameters():
            p.requires_grad = False
        self.update_network = self.update_net_class(in_channels=self.num_channels, out_channels=self.update_net_channel_dims, bias=self.bias)
        for p in self.update_network.parameters():
            p.requires_grad = False
        self.normalise = self.config.get('normalise')
        self.replace = self.config.get('replace')
        self.debugging = self.config.get('debugging')
        self.tanh = torch.nn.Tanh()

            
    def alive(self, x):  # return maxpool over the alive channel (1,1,:,:), used to zero-out cells who have no surrounding cell with alive channel above alive thereshold
        return F.max_pool3d(x[:, self.living_channel_dim, :, :, :], kernel_size=3, stride=1, padding=1)
    
    def perceive(self, x):
        return self.perception_net(x)

    def update(self, x):
        alive_thresdhold =  self.alpha_living_threshold * self.alive(x).max()
        if torch.isnan(alive_thresdhold): alive_thresdhold = np.NINF
        pre_life_mask = self.alive(x) > alive_thresdhold

        out = self.perceive(x)
        out = self.update_network(out)
        
        if self.debugging:
            for layer in list(self.perception_net.parameters()):
                print(f"perception_net layer weight max: {layer.max()}, min: {layer.min()}")
            for layer in list(self.update_network.parameters()):
                print(f"update_network layer weight max: {layer.max()}, min: {layer.min()}")
            print(f"x max: {x.max()}")
            print(f"out max: {out.max()}")

        if self.normalise: 
            if self.replace:
                x = out / out.max()
            else:
                x = x + out / out.max()
        else:   
            if self.replace:
                x = out 
            else:
                x = x + out    
            
        if x.max() > 100000:
            raise ValueError('NCA states are exploding')
        
        
        alive_thresdhold =  self.alpha_living_threshold * self.alive(x).max()
        if torch.isnan(alive_thresdhold): alive_thresdhold = np.NINF
        post_life_mask = self.alive(x) > alive_thresdhold
        
        life_mask = (pre_life_mask & post_life_mask) 
        x = x * life_mask   # Zero-out cells who have no surrounding cell with alive channel above alive thereshold
        
        if self.debugging:
            print(f"\npre_life_mask {torch.sum(pre_life_mask)}")
            print(f"post_life_mask {torch.sum(post_life_mask)}")
            print(f"life_mask {torch.sum(life_mask)}")
        
        return x, life_mask

    def forward(self, x, steps, reading_channel, run_pca=False, visualise_weights=False, visualise_network=False, inOutdim=None):
        if visualise_weights:
            from celluloid import Camera
            fig2, ax2 = pyplot.subplots(3)
            camera_layers = Camera(fig2)
            
            # Delta
            # fig3, ax3 = pyplot.subplots(3)
            # camera_layers_delta = Camera(fig3)
            
        elif visualise_network:
            from celluloid import Camera
            fig1 = pyplot.figure(figsize=(10,8))
            cameraNetwork = Camera(fig1)
        
        weights_for_pca = [] if run_pca else None
        for step in range(steps):
            if visualise_network:   # Only generate network visualation at a time
                x_ = x.clone()
                x_[:,:,-1,inOutdim[1]:,:] = 0.0 
                cameraNetwork = visualiseNetwork(x[0][reading_channel], inOutdim, cameraNetwork, animated=True)
            elif visualise_weights:  # Only generate voxel visualation at a time
                x_ = x.clone()
                x_[:,:,-1,inOutdim[1]:,:] = 0.0 
                # cameraVoxels = visualiseVoxs(np.abs(x[0][reading_channel].numpy())/(10*np.max(np.abs(x[0][reading_channel].numpy()))), cameraVoxels, ax2)
                camera_layers = visualiseVoxs2Dmulti(x_, camera_layers, fig2, ax2, step, None)
            
            x, life_mask = self.update(x)
            x[:,:,-1,inOutdim[1]:,:] = 0.0 
            
            if run_pca:
                x_ = x.clone()
                x_[:,:,-1,inOutdim[1]:,:] = 0.0 
                weights_for_pca.append(x_[0][reading_channel].flatten().detach().numpy())
                
            # # Delta
            # if visualise_weights:
            #     camera_layers_delta = visualiseVoxs2Dmulti(abs(x-x_), camera_layers_delta, fig3, ax3, step, 'gray')

        if visualise_weights:
            id_ = str(int(time.time()))
            animationVoxels = camera_layers.animate() 
            # animationVoxels = cameraVoxels.animate() 
            # animationVoxels.save('animation_weights_' + id_ + '.mp4', fps=2, dpi=300)
            pyplot.show()
            
            # # Delta
            # camera_layers_delta = camera_layers_delta.animate() 
            # pyplot.show()
        elif visualise_network:
            id_ = str(int(time.time()))
            animationNetwork = cameraNetwork.animate() 
            # animationNetwork.save('animation_netwok_' + id_ + '.mp4', fps=2, dpi=300)
            pyplot.show()
    
        return x, weights_for_pca
    

