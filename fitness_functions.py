import gym
from gym import wrappers as w
from gym.spaces import Discrete, Box
import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import time

from policies import MLPn
from utils_and_wrappers import FireEpisodicLifeEnv, ScaledFloatFrame

from utils_and_wrappers import generate_seeds3D, policy_layers_parameters, dimensions_env
from NCA_3D import CellCAModel3D


gym.logger.set_level(40)
torch.set_default_dtype(torch.float64)



def fitnessRL(evolved_parameters, nca_config, render = False, debugging=False, visualise_weights=False, visualise_network = False): 
    """
    Returns the NEGATIVE episodic fitness of the agents.
    """

    with torch.no_grad():
        
        cum_reward = 0
        seed_offset = 0
        for i, environment in enumerate(nca_config['environment']):
                        
            # Load environment
            try:
                env = gym.make(environment, verbose = 0)
            except:
                env = gym.make(environment)
                
            if not nca_config['random_seed_env']: 
                env.seed(nca_config['RANDOM_SEED'])
            
            if environment[-12:-6] == 'Bullet' and render:
                    env.render()  # bullet envs
            mujoco_env = False
        
            # For environments with several intra-episode lives -eg. Breakout-
            try: 
                if 'FIRE' in env.unwrapped.get_action_meanings():
                    env = FireEpisodicLifeEnv(env)
            except: 
                pass
            
            

            # Check if selected env is pixel or state-vector and its dimensions
            input_dim, action_dim, pixel_env = dimensions_env(environment)
            if pixel_env == True: 
                env = w.ResizeObservation(env, 32)        # Resize and normilise input   
                env = ScaledFloatFrame(env)

            if nca_config['NCA_dimension'] == 2:
                raise NotImplementedError
            elif nca_config['NCA_dimension'] == 3:
                p = MLPn(input_space=nca_config['size_substrate'], action_space=action_dim, hidden_dim=nca_config['size_substrate'], bias=False, layers=nca_config['policy_layers']) 
            
            for param in p.parameters():
                param.requires_grad = False
                
            
            # Initilise NCA with config dict
            if nca_config['NCA_dimension'] == 2:
                raise NotImplementedError
            elif nca_config['NCA_dimension'] == 3:
                if nca_config['NCA_MLP']:
                    raise NotImplementedError
                else:
                    ca = CellCAModel3D(nca_config)            
            nca_nb_weights = torch.nn.utils.parameters_to_vector(ca.parameters()).shape[0] 
            
            if render:
                nca_nb_weights = torch.nn.utils.parameters_to_vector(ca.parameters()).shape[0]
                seed_size = nca_config['NCA_channels']*nca_config['policy_layers']*nca_config['size_substrate']*nca_config['size_substrate']
                policy_nb_functional_params = torch.nn.utils.parameters_to_vector(p.parameters()).shape[0]
                print(f'Policy has: {policy_nb_functional_params}')
                print('\n.......................................................')
                print('\n' + str(nca_config['environment']) + ' with', nca_nb_weights, 'trainable parameters controlling a policy', str(p)[:3], 'with', policy_nb_functional_params, 'effective weights with a seed size of', seed_size, ' and seed type', nca_config['seed_type'], '\n')
                if nca_config['plastic']: print('Plastic Policy network') 
                print('.......................................................\n')
            
            # Load evolved weights into the NCA
            nn.utils.vector_to_parameters( torch.tensor (evolved_parameters[:nca_nb_weights], dtype=torch.float64 ),  ca.parameters() )
                
                
            observation = env.reset().astype(np.float64)
            if pixel_env: observation = observation.flatten()

            # Load or generate (if random) seed
            # Generate random episode seed
            if nca_config['random_seed']:
                if nca_config['NCA_dimension'] == 2:
                    raise NotImplementedError
                elif nca_config['NCA_dimension'] == 3:
                    seed = generate_seeds3D(policy_layers_parameters(p), nca_config['seed_type'][i], nca_config['NCA_channels'], observation, environment)
            
            # Load co-evolved seed
            elif nca_config['co_evolve_seed']:
                sp = nca_config['seeds_shapes'][i]
                evolved_seed = torch.tensor(evolved_parameters[nca_nb_weights + seed_offset : nca_nb_weights + seed_offset + nca_config['seeds_size'][i]])
                if nca_config['NCA_dimension'] == 2: 
                    raise NotImplementedError
                if nca_config['NCA_dimension'] == 3:
                    seed = torch.reshape(evolved_seed, sp[0])
                    
                seed_offset += nca_config['seeds_size'][i]
            
            # Load fix seed
            else:
                seed = nca_config['seeds'][i]
            
            # Generate policy networks with the NCA
            if not nca_config['plastic']:
                
                (a, b, c) = (0, 1, 2) if not pixel_env else (2, 3, 4)
                
                if nca_config['NCA_dimension'] == 2:
                    raise NotImplementedError
                            
                elif nca_config['NCA_dimension'] == 3:
                    
                    if nca_config['NCA_MLP']:
                        raise NotImplementedError
                    else:                    
                        new_pattern, _weights_for_pca_ = ca.forward(seed, steps=nca_config['NCA_steps'], reading_channel=nca_config['reading_channel'], policy_layers = nca_config['policy_layers'], run_pca=False, visualise_weights=visualise_weights, visualise_network=visualise_network, inOutdim=[input_dim,action_dim])
                        generated_policy_weights = new_pattern.detach()[0]
                        
                    # Load generated weights into policy network 
                    reading_channel = nca_config['reading_channel']
                    for i in range(nca_config['policy_layers']):
                        if i == nca_config['policy_layers'] - 1: # last layer of the policy
                            p.out[2*i].weight = nn.Parameter(generated_policy_weights[reading_channel][i][:action_dim,:], requires_grad=False) 
                        else:
                            p.out[2*i].weight = nn.Parameter(generated_policy_weights[reading_channel][i], requires_grad=False) 
                    
                    if nca_config['NCA_MLP']:
                        
                        raise NotImplementedError
                        
                        
                if debugging:
                    
                    for layer in list(ca.parameters()):
                        print(f"NCA layer weight max: {layer.max()}, min: {layer.min()}")
                    print(f"Policy weights max: {generated_policy_weights.max()}, min: {generated_policy_weights.min()}, mean: {generated_policy_weights.mean()}")
                    
                        
                    
                penalty = 0
                if nca_config['penalty_off_topology']:                    
                    raise NotImplementedError
                    
                    
                # Prevents runnning environment in case the NCA pattern has died
                if torch.sum(abs(generated_policy_weights)) == 0:
                    if render:
                        print('\nThe NCA produce an empty pattern, skipping simulation of the environment.\n')
                    return np.Inf

            penalty = 0
            
            # Burnout phase for the bullet quadruped so it starts off from the floor
            if 'AntBullet' in environment:
                action = np.zeros(8)
                for _ in range(40):
                    __ = env.step(action)        
                    
            # Inner loop
            dim_first_fc = p.out[0].weight.shape[1]
            neg_count = 0
            rew_ep = 0
            t = 0
            while True:
                
                # Generate and load policy networks with the NCA
                if nca_config['plastic']:
                    
                    raise NotImplementedError
                        
                # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
                if isinstance(env.observation_space, Discrete): 
                    observation = (observation == torch.arange(env.observation_space.n))

                
                o3 = p(torch.tensor(observation))
                
                # # Bounding the action space
                if environment == 'CarRacing-v0':
                    action = np.array([ torch.tanh(o3[0]), torch.sigmoid(o3[1]), torch.sigmoid(o3[2]) ]) 
                    o3 = o3.numpy()
                elif 'Bullet' in environment or str(env.action_space)[0:14] == 'Box(-1.0, 1.0,' or mujoco_env:
                    o3 = np.tanh(o3).numpy()
                    action = o3
                else: 
                    if isinstance(env.action_space, Box):
                        action = o3.numpy()                         
                        action = np.clip(action, env.action_space.low, env.action_space.high)  
                    elif isinstance(env.action_space, Discrete):
                        action = np.argmax(o3).numpy()
        
                if debugging:
                # if True:
                    print(o3)
                    print(action)
                    print('\n')
                    
                # Environment simulation step
                observation, reward, done, info = env.step(action)  
                if 'AntBullet' in environment: reward = env.unwrapped.rewards[1] # Distance walkel
                rew_ep += reward
                
                if environment[-12:-6] != 'Bullet' and render:
                    env.render('human') # Gym envs
                if render:
                    time.sleep(0.005)
                
                if pixel_env: observation = observation.flatten()
                observation = observation.astype(np.float64)
                                    
                # Early stopping conditions
                if t > 100:
                    if mujoco_env:
                        neg_count = neg_count+1 if reward < 1.01 else 0
                    else:
                        neg_count = neg_count+1 if reward < 0.01 else 0
                    
                    if environment == 'CarRacing-v0':
                        if (done or neg_count > 20):
                            break
                    elif 'Bullet' in environment:
                        if (done or neg_count > 30):
                            break
                    else:
                        if done or neg_count > 50:
                            break
                t += 1
                
            env.close()
            
            cum_reward += (rew_ep - penalty)
            
            if render:
                print(f"{environment} reward without penalty: {rew_ep}")
        
        
    if debugging or render or visualise_weights or visualise_network:
        print(f"\nEpisode cumulative reward {cum_reward}\n")
        
    return -cum_reward




def evaluate(argv):
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--id', type=str, default='1645447353', metavar='', help='Run id')   # lander 5 layers
    # parser.add_argument('--id', type=str, default='1646940683', metavar='', help='Run id')   # lander single seed
    # parser.add_argument('--id', type=str, default='1645360631', metavar='', help='Run id')   # ant 3 layers
    # parser.add_argument('--id', type=str, default='1645605120', metavar='', help='Run id')   # ant 30 layers deep one
    # parser.add_argument('--id', type=str, default='1647084085', metavar='', help='Run id')   # ant single seed
    parser.add_argument('--id', type=str, default='1644785913', metavar='', help='Run id')    # metamorphosis quadrupeds
    
    parser.add_argument('--render', type=bool, default=1)
    parser.add_argument('--visualise_weigths', type=bool, default=0) 
    parser.add_argument('--visualise_network', type=bool, default=0)
    parser.add_argument('--mean_solution', type=bool, default=1,  help='Whether to use the best population mean, else it will use best individual solution')
    parser.add_argument('--evaluation_runs', type=int, default=1,  help='Number of runs to evaluate model')


    args = parser.parse_args()
    
    if args.visualise_weigths and args.visualise_network:
        raise ValueError('Can not visualise both weight matrix and network at the same time')
    
    # Load NCA config
    nca_config = pickle.load( open( 'saved_models/' + args.id + '/nca.config', "rb" ) )   
    
    for key, value in nca_config.items():
        if key != 'seeds':
            print(key,':',value)
    
    # Load evolved NCA weightsl
    for root, dirs, files in os.walk("saved_models/" + args.id):
        for file in files:
            if args.mean_solution and file and 'meansolution' in file:
                evolved_parameters = np.load('saved_models/' + args.id + '/' + file) 
                print(f"\nUsing best MEAN solution") 
            elif not args.mean_solution and file and 'bestsolution' in file:
                evolved_parameters = np.load('saved_models/' + args.id + '/' + file) 
                print(f"\nUsing BEST individual solution") 

    evals = []
    runs = args.evaluation_runs
    for _ in range(runs):
        evals.append(-1*fitnessRL(evolved_parameters=evolved_parameters, nca_config=nca_config, render=args.render, visualise_weights=args.visualise_weigths, visualise_network=args.visualise_network))
    evals = np.array(evals)
    print(f'mean reward {np.mean(evals)}. Var: {np.std(evals)}. Shape {evals.shape}')

    
if __name__ == '__main__':
    import argparse
    import torch
    import sys
    evaluate(sys.argv)