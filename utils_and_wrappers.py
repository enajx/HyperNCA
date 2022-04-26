import gym
import numpy as np
import torch
from matplotlib import pyplot
torch.set_default_dtype(torch.float64)


def generate_seeds3D(policy_layers, seed_type, channels, activations = None, environment=None): 

    layers = len(policy_layers)
    max_dim = 0
    for i, layer in enumerate(policy_layers):
        if max(layer['in_dim'],layer['out_dim']) > max_dim:
            max_dim = max(layer['in_dim'],layer['out_dim'])
        if i == layers - 1: # last layer
            out_dim = layer['out_dim']

    if seed_type == 'randomU':
        s = torch.rand(1, channels, layers, max_dim, max_dim)  # batch size, channels, y_dim, x_dim
    elif seed_type == 'randomU2':
        s = 2*torch.rand(1, channels,layers, max_dim, max_dim) - 1
    elif seed_type == 'randomU3':
        s = 2*torch.rand(1, channels,layers,max_dim, max_dim) - 1
        s /= 2
    elif seed_type == 'randomU4':
        s = torch.rand(1, channels,layers, max_dim, max_dim)
        s /= 10
    elif seed_type == 'randomU5':
        s = 2*torch.rand(1, channels,layers, max_dim, max_dim) - 1
        s /= 10
    elif seed_type == 'randomN':
        s = torch.randn(1, channels,layers, max_dim, max_dim)
    elif seed_type == 'ones':
        s = torch.ones(1, channels, layers,max_dim, max_dim)
    elif seed_type == 'zeros':
        s = torch.zeros(1, channels, layers,max_dim, max_dim)
    elif seed_type == 'single_seed':
        s = torch.zeros(1, channels, layers,max_dim, max_dim)
        if 'Lander' in environment:
            s[:, :, :layers-1, max_dim//2,  max_dim//2] = 1.0   # lunar lander
        elif 'Ant' in environment:
            s[:, :, :, max_dim//2,  max_dim//2] = 1.0   # ant
    elif seed_type == 'single_seed_big':
        s = torch.zeros(1, channels, layers,max_dim, max_dim)
        s[:, :, :, max_dim//2-1:max_dim//2+1,  max_dim//2-1:max_dim//2+1] = 1.0
    elif seed_type == 'two_seeds':
        s = torch.zeros(1, channels, layers,max_dim, max_dim)
        s[:, :, :, max_dim//2,  max_dim//3] = 1.0
        s[:, :, :, max_dim//2,  max_dim - max_dim//3] = 1.0
    elif seed_type == 'three_seeds':
        s = torch.zeros(1, channels, layers,max_dim, max_dim)
        s[:, :, :, max_dim//2,  max_dim//4] = 1.0
        s[:, :, :, max_dim//2,  max_dim//2] = 1.0
        s[:, :, :, max_dim//2,  max_dim - max_dim//4] = 1.0
    elif seed_type == '1single_seed':
        s = torch.zeros(1, channels, layers,max_dim, max_dim)
        # s[:, 0, :, :,  :] = 1.0
        s[:, 0, :, max_dim//2,  max_dim//2] = 1.0
    elif seed_type == 'four_seeds':
        s = torch.zeros(1, channels, layers,max_dim, max_dim)
        s[:, :,:, max_dim - max_dim//4, max_dim - max_dim//3] = 1.0
        s[:, :,:, max_dim//4, max_dim - max_dim//3] = 1.0
        s[:, :,:, max_dim - max_dim//4, max_dim//3] = 1.0
        s[:, :,:, max_dim//4, max_dim//3] = 1.0
    elif seed_type == '-ones':
        s = -1.0*torch.ones(1, channels, layers, max_dim, max_dim)
    elif seed_type == 'activations':
        s = torch.zeros(1, channels, layers,max_dim, max_dim)
        n = activations.shape[0]
        s[:, :, : , max_dim//2, max_dim//2 - n//2 : max_dim//2 + n//2] = torch.tensor(activations)
    else:
        raise AssertionError

    s[:,:,-1,out_dim:,:] = 0.0
    return s # batch, channels, depth (num layers), ydim, xdim


def policy_layers_parameters(p):
    policy_layers = []
    for name, layer in p.named_modules():
        if isinstance(layer, torch.nn.Linear):
            l = {
                'layer' : name,
                'in_dim' : layer.in_features,
                'out_dim': layer.out_features,
                'bias' : layer.bias
            } 
            policy_layers.append(l)
        elif isinstance(layer, torch.nn.Conv2d):
            l = {
                'layer' : name,
                'in_dim' : layer.in_channels,
                'out_dim': layer.out_channels,
                'kernel_size' : layer.kernel_size,
                'stride' : layer.stride,
                'bias' : layer.bias
            } 
            policy_layers.append(l)
        elif isinstance(layer, torch.nn.Conv3d):
            l = {
                'layer' : name,
                'in_dim' : layer.in_channels,
                'out_dim': layer.out_channels,
                'kernel_size' : layer.kernel_size,
                'stride' : layer.stride,
                'bias' : layer.bias
            } 
            policy_layers.append(l)
    
    return policy_layers



def plots_rewads_save(id_, rewards_mean, rewards_best):     
    pyplot.plot(rewards_mean, label="Mean")
    pyplot.plot(rewards_best, label="Best")
    pyplot.xlabel('Generation', fontsize=12)
    pyplot.ylabel('Negative reward', fontsize=12)
    pyplot.legend(loc="best", prop={'size': 14})
    pyplot.savefig('saved_models_metamorphosis' + "/"+ id_ + '/rewards_neg.pdf', dpi=300)
    pyplot.clf()
    pyplot.plot(-1*rewards_mean, label="Mean")
    pyplot.plot(-1*rewards_best, label="Best")
    pyplot.xlabel('Generation', fontsize=12)
    pyplot.ylabel('Reward', fontsize=12)
    pyplot.legend(loc="best", prop={'size': 14})
    pyplot.savefig('saved_models_metamorphosis' + "/"+ id_ + '/rewards.pdf', dpi=300)
    pyplot.clf()
    pyplot.close()
    

def visuvisualise_plastic_weights(p, axes, seed, rows, camera, camera2):
        
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(seed[0 , i%rows]) 
        ax.axis('off')
    camera.snap()
    
    _1, _2, _3 = list(p.parameters())          #MLP          
    ws = [_1, _2, _3]
    ws = [w.detach().numpy() for w in ws]
    
    ws[0] = ws[0]
    ws[1] = ws[1].T
    ws[2] = ws[2].T

    
    layer_names = ['FC layer 1', 'FC layer 2', 'FC layer 3'] 
    
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(ws[i])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel(layer_names[i], fontsize=14)
    
    camera2.snap()


def visualise_plastic_weights_plastic_inFitness(axes , camera, camera2):

    animation = camera.animate() 
    animation.save('animation.mp4', dpi=300)
    
    animation2 = camera2.animate() 
    #% start: automatic generated code from pylustrator
    pyplot.figure(2).ax_dict = {ax.get_label(): ax for ax in pyplot.figure(2).axes}
    import matplotlib as mpl
    pyplot.figure(2).axes[0].set_position([0.154127, 0.110000, 0.072187, 0.770000])
    pyplot.figure(2).axes[1].xaxis.labelpad = 10.720000
    pyplot.figure(2).axes[2].set_position([0.817358, 0.110000, 0.036094, 0.770000])
    pyplot.figure(2).ax_dict = {ax.get_label(): ax for ax in pyplot.figure(2).axes}
    pyplot.figure(2).axes[1].set_position([0.366029, 0.278873, 0.299191, 0.398921])
    #% end: automatic generated code from pylustrator
    animation2.save('animation_2.mp4', dpi=300)
    pyplot.show()    

def visualise_plastic_weights_NonPlastic_inFitness(p, axes, axes2, fig2, camera, camera2):
    
    _1, _2, _3 = list(p.parameters())          #MLP          
    ws = [_1, _2, _3]
    ws = [w.detach().numpy() for w in ws]
    
    ws[0] = ws[0]
    ws[1] = ws[1].T
    ws[2] = ws[2].T

    
    layer_names = ['FC layer 1', 'FC layer 2', 'FC layer 3']
    
    for i, ax in enumerate(axes2.flat):
        im = ax.imshow(ws[i])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel(layer_names[i], fontsize=12)

    fig2.colorbar(im, ax=axes2.flat,  orientation="horizontal", pad=0.2)
    pyplot.figure(1).ax_dict = {ax.get_label(): ax for ax in pyplot.figure(1).axes}
    import matplotlib as mpl
    pyplot.figure(1).axes[0].set_position([0.077188, 0.192000, 0.132839, 0.708477])
    pyplot.figure(1).axes[2].set_position([0.793879, 0.192000, 0.066420, 0.708477])
    pyplot.figure(1).ax_dict = {ax.get_label(): ax for ax in pyplot.figure(1).axes}
    pyplot.figure(1).ax_dict["<colorbar>"].set_position([0.177500, 0.058000, 0.653125, 0.043542])
    pyplot.figure(1).ax_dict = {ax.get_label(): ax for ax in pyplot.figure(1).axes}
    pyplot.figure(1).axes[1].set_position([0.316654, 0.270333, 0.385441, 0.513922])
    pyplot.show()


def plot_index_weights(fc):
    
    fig, ax = pyplot.subplots()
    im = ax.imshow(fc.weight)

    # Loop over data dimensions and create text annotations.
    for i in range(fc.weight.shape[0]):
        for j in range(fc.weight.shape[0]):
            text = ax.text(j, i, str(i)+','+str(j),
                        ha="center", va="center", color="w", fontsize='xx-small')

    ax.set_title("Weight fc")
    fig.tight_layout()
    pyplot.show()


# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float64)

    def observation(self, observation):
        # This undoes the memory optimization, use with smaller replay buffers only.
        return np.array(observation).astype(np.float64) / 255.0

class FireEpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Check current lives, make loss of life terminal then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()

        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            # done = True
            obs, _, done, _ = self.env.step(1)
            if done:
                self.env.reset()
            obs, _, done, _ = self.env.step(2)
            if done:
                self.env.reset()

        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


def calc_fan(weight_shape):
    """
    Source: https://github.com/ddbourgin/numpy-ml/
    Compute the fan-in and fan-out for a weight matrix/volume.
    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume. The final 2 entries must be
        `in_ch`, `out_ch`.
    Returns
    -------
    fan_in : int
        The number of input units in the weight tensor
    fan_out : int
        The number of output units in the weight tensor
    """
    if len(weight_shape) == 2:
        fan_in, fan_out = weight_shape
    elif len(weight_shape) in [3, 4]:
        in_ch, out_ch = weight_shape[-2:]
        kernel_size = np.prod(weight_shape[:-2])
        fan_in, fan_out = in_ch * kernel_size, out_ch * kernel_size
    else:
        raise ValueError("Unrecognized weight dimension: {}".format(weight_shape))
    return fan_in, fan_out


def xavier_glorot_uniform(weight_shape, gain=1.0):
    """
    Source: https://github.com/ddbourgin/numpy-ml/
    Initialize network weights `W` using the Glorot uniform initialization
    strategy.
    For tanh
    -----
    The Glorot uniform initialization strategy initializes weights using draws
    from ``Uniform(-b, b)`` where:
    .. math::
        b = \\text{gain} \sqrt{\\frac{6}{\\text{fan_in} + \\text{fan_out}}}
    The motivation for Glorot uniform initialization is to choose weights to
    ensure that the variance of the layer outputs are approximately equal to
    the variance of its inputs.
    This initialization strategy was primarily developed for deep networks with
    tanh and logistic sigmoid nonlinearities.
    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.
    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    b = gain * np.sqrt(6 / (fan_in + fan_out))
    return np.squeeze(np.random.uniform(-b, b, size=weight_shape))


def kaiming_he_uniform(weight_shape):
    """
    Source: https://github.com/ddbourgin/numpy-ml/
    Initializes network weights `W` with using the He uniform initialization
    strategy.
    For relu
    -----
    The He uniform initializations trategy initializes thew eights in `W` using
    draws from Uniform(-b, b) where
    .. math::
        b = \sqrt{\\frac{6}{\\text{fan_in}}}
    Developed for deep networks with ReLU nonlinearities.
    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.
    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    b = np.sqrt(6 / fan_in)
    return np.squeeze(np.random.uniform(-b, b, size=weight_shape))


def dimensions_env(environment):
    """
    Look up observation and action space dimension
    """
    from gym.spaces import Discrete, Box
    
    env = gym.make(environment)    
    if len(env.observation_space.shape) == 3:     # Pixel-based environment
        pixel_env = True
        input_dim = 32*32*3
    elif len(env.observation_space.shape) == 1:   # State-based environment 
        pixel_env = False
        input_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, Discrete):
        pixel_env = False
        input_dim = env.observation_space.n
    else:
        raise ValueError('Observation space not supported')

    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    else:
        raise ValueError('Action space not supported')
    

    return input_dim, action_dim, pixel_env