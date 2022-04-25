# Code adapted from https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network
from matplotlib import pyplot
from matplotlib.transforms import IdentityTransform
from math import cos, sin, atan
import torch
import numpy as np
from collections import OrderedDict
from celluloid import Camera


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius, idx_layer, idx_neuron, nb_layers):
        if (idx_layer == 0 and idx_neuron < inDim) or (idx_layer == nb_layers - 1):
            circle = pyplot.Circle((self.y, self.x), radius=neuron_radius, fill=True, color='black')
        else:
            circle = pyplot.Circle((self.y, self.x), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, layer_weights):
        self.vertical_distance_between_layers = 25 
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.layer_weights = layer_weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, weight):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D( (neuron1.y - y_adjustment, neuron2.y + y_adjustment), (neuron1.x - x_adjustment, neuron2.x + x_adjustment), color='k', alpha=min(weight, 1), linewidth=weight**2) 
        pyplot.gca().add_line(line)

    def draw(self, nb_layers, layerType=0 ):
        for i, neuron in enumerate(reversed(self.neurons)):
            neuron.draw( self.neuron_radius, layerType, i, nb_layers)
            if self.previous_layer:
                for j, previous_layer_neuron in enumerate(reversed(self.previous_layer.neurons)):
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, self.layer_weights[j,i])
        

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer, weights):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0
        self.weights = weights

    def add_layer(self, number_of_neurons, layer_weights ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, layer_weights)
        self.layers.append(layer)

    def draw(self, camera):
        
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            # if i == len(self.layers)-1:
                # i = -1
            layer.draw( nb_layers= len(self.layers), layerType=i)
        pyplot.axis('scaled')
        pyplot.axis('off')
        
        if camera is not None:
            camera.snap()

class DrawNN():
    def __init__( self, neural_network, weights, animated, mode):
        self.neural_network = neural_network
        self.weights = weights
        self.mode = mode
        self.animated = animated

    def draw(self, camera):
        widest_layer = max( self.neural_network)
        network = NeuralNetwork( widest_layer, self.weights)
        for i, l in enumerate(self.neural_network):
            network.add_layer(l, self.weights[i])
        network.draw(camera)
        




def visualiseNetwork(policy_weights, inOutdim, camera, animated=False, mode = 'magnitude'):
    
    layers_weights = []
    if isinstance(policy_weights, OrderedDict):
        fc1 = abs(policy_weights['fc1.weight'].numpy())
        fc1 = np.swapaxes(fc1, 0, 1)
        fc2 = abs(policy_weights['fc2.weight'].numpy())
        fc2 = np.swapaxes(fc2, 0, 1)
        fc3 = abs(policy_weights['fc3.weight'].numpy())
        fc3 = np.swapaxes(fc3, 0, 1)
        max_abs_weight = max(fc1.max(), fc2.max(), fc3.max())
    elif isinstance(policy_weights, torch.Tensor) or isinstance(policy_weights, list):
        for i in range(policy_weights.shape[0]):
            fc = policy_weights[i].numpy()
            layers_weights.append(np.swapaxes(abs(fc), 0, 1))
            

    max_abs_weight = 0
    for i in range(policy_weights.shape[0]):
        max_layer = policy_weights[i].numpy().max()
        if max_layer > max_abs_weight:
            max_abs_weight = max_layer
    
    for i in range(policy_weights.shape[0]):
        layers_weights[i] /= max_abs_weight
    
        
    global inDim
    inDim = inOutdim[0]
    global outDim
    outDim = inOutdim[1]
    
    
    layers_sizes = [layer.shape[0] for layer in layers_weights]
    layers_sizes.append(outDim)
    layers_weights = [None] + layers_weights
    network = DrawNN(layers_sizes, layers_weights, animated, mode)  
    network.draw(camera)                                                                                      

    return camera

def visualisePerceptron(policy_weights, inOutdim, camera, animated=False, mode = 'magnitude'):
    
    if isinstance(policy_weights, OrderedDict):
        fc1 = abs(policy_weights['fc1.weight'].numpy())
        fc1 = np.swapaxes(fc1, 0, 1)
        max_abs_weight = max(fc1.max())
    elif isinstance(policy_weights, torch.Tensor) or isinstance(policy_weights, list):
        if isinstance(policy_weights[0], torch.Tensor):
            fc1 = abs(policy_weights[0].numpy()) 
        else:
            fc1 = abs(policy_weights[0]) 
        fc1 = np.swapaxes(fc1, 0, 1)       
        max_abs_weight = fc1.max()
    
    if max_abs_weight != 0:
        fc1 /= max_abs_weight
            
        
    global inDim
    inDim = inOutdim[0]
    global outDim
    outDim = inOutdim[1]
    
    weights = [None, fc1] 
    network = DrawNN([fc1.shape[0],fc1.shape[1]], weights, animated, mode)   
    network.draw(camera)                                                                                      

    return camera

    


def cuboid_data(pos, size=(1,1,1)):
    # Code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(alpha, pos=(0,0,0), ax=None):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos )
        ax.plot_surface(X, Y, Z, color='black', rstride=1, cstride=1, alpha=alpha)

def plotMatrix(ax, matrix):
    # plot a Matrix 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                # if matrix[i,j,k] > 0.001:
                if True:
                    plotCubeAt(alpha=matrix[i,j,k], pos=(i-0.5,j-0.5,k-0.5), ax=ax)            


def visualiseVoxs(data, camera, ax):

    x = np.array([-2, -1, 0, 1])
    my_xticks = [None,'Input layer','Hidden Layer','Output layer']
    pyplot.xticks(x, my_xticks)

    ax.set_box_aspect(aspect = (0.5,1,1))

    plotMatrix(ax, data)

    if camera is not None:
        camera.snap()
    
    return camera
    
def visualiseVoxs2D(data, camera, axes):

    reading_channel = 0
    
    im = axes.imshow(data[reading_channel])
    axes.set_ylabel('Layer ' + str(1), rotation=0, labelpad=20, fontdict={'size':12})
    
    if camera is not None:
        camera.snap()
    
    return camera
    
def visualiseVoxs2Dmulti(data, camera, fig, axes, step, colors):

    reading_channel = 0
    outDim = 8 # ada
    
    for i in range(data[0][reading_channel].shape[0]):
        im = axes[i].imshow(data[0][reading_channel][i], cmap=colors)
        axes[i].set_ylabel('Layer ' + str(i+1), rotation=0, labelpad=20, fontdict={'size':12})

    print(f"\n") 
    print(f"\n") 
    pyplot.text(0.8, 0.1, 'Step ' + str(step), fontsize=12, transform=fig.transFigure)
    
    if camera is not None:
        camera.snap()
    
    return camera
    
    
    
def visualiseVoxs2DmultiMLP(data, camera, fig, axes, reading_channel=0, step=-1):
    # pyplot.colorbar(im)
    
    for i in range(data[reading_channel].shape[0]):
        im = axes[i].imshow(data[reading_channel][i])
        axes[i].set_ylabel('Layer ' + str(i+1), rotation=0, labelpad=20, fontdict={'size':12})

    pyplot.text(0.8, 0.1, 'Step ' + str(step), fontsize=12, transform=fig.transFigure)

    if camera is not None:
        camera.snap()
    
    return camera
    
    
    
