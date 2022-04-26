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
from matplotlib import pyplot

from policies import MLPn
from utils_and_wrappers import FireEpisodicLifeEnv, ScaledFloatFrame, dimensions_env
from NCA_3D import CellCAModel3D

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
gym.logger.set_level(40)


def fitnessRL(evolved_parameters, nca_config, render = False, debugging=False, visualise_weights=False, visualise_network = False, run_PCA = False, path=False): 
    """
    Returns the NEGATIVE episodic fitness of the agents.
    """

    with torch.no_grad():
        
        rewards = []
        weights_for_pca = [] 
        metamorph_times = []
        for i, environment in enumerate(nca_config['environment']):
                        
            env = gym.make(environment)
                
            if not nca_config['random_seed_env']: 
                env.seed(nca_config['RANDOM_SEED'])    
                torch.manual_seed(nca_config['RANDOM_SEED'])
                np.random.seed(nca_config['RANDOM_SEED'])
            
            if 'Bullet' in environment and render:
                    env.render()  # bullet envs

            # Is a mujoco env?        
            try:
                mujoco_env = 'mujoco' in str(env.env.model)
            except:
                mujoco_env = False
        
            try: 
                if 'FIRE' in env.unwrapped.get_action_meanings():
                    env = FireEpisodicLifeEnv(env)
            except: 
                pass

            # Check if selected env is pixel or state-vector and its dimensions
            input_dim, action_dim, pixel_env = dimensions_env(environment)
            if pixel_env == True: 
                env = w.ResizeObservation(env, 84)        # Resize and normilise input   
                env = ScaledFloatFrame(env)

            # Initialise policy network just so we know how many parameters it has
            if pixel_env == True:   
                raise NotImplementedError
            else:
                if nca_config['NCA_dimension'] == 2:
                    raise NotImplementedError
                elif nca_config['NCA_dimension'] == 3:
                    p = MLPn(input_space=input_dim, action_space=action_dim, hidden_dim=nca_config['size_substrate'], bias=False, layers=nca_config['policy_layers']) 
                    
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
                policy_nb_functional_params = torch.nn.utils.parameters_to_vector(p.parameters()).shape[0]
                seed_size = nca_config['NCA_channels']*nca_config['policy_layers']*nca_config['size_substrate']*nca_config['size_substrate']
                print('\n.......................................................')
                print('\n' + str(nca_config['environment']) + ' with', nca_nb_weights, 'trainable parameters controlling a policy', str(p)[:3], 'with', policy_nb_functional_params, 'effective weights with a seed size of', seed_size, ' and seed type', nca_config['seed_type'], '\n')
                if nca_config['plastic']: print('Plastic Policy network') 
                print('.......................................................\n')
            
            # Load evolved weights into the NCA
            nn.utils.vector_to_parameters( torch.tensor (evolved_parameters[:nca_nb_weights], dtype=torch.float64 ),  ca.parameters() )
                
                
            observation = env.reset() 
            if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)       ()

            # Load initial seed or developped seed if not first environmnet
            if i == 0:
                seed = nca_config['seed']
                nca_steps = nca_config['NCA_steps']
            else: 
                nca_steps = nca_config['metamorphosis_NCA_steps']
                if nca_config['NCA_MLP']:
                    seed = new_pattern.copy()
                else:
                    seed = new_pattern.clone().detach()

            
            # Generate policy networks with the NCA
            
            (a, b, c) = (0, 1, 2) if not pixel_env else (2, 3, 4)
        
            if nca_config['NCA_dimension'] != 3:
                raise NotImplementedError
            
            if nca_config['NCA_MLP']:
                raise NotImplementedError

            new_pattern, _weights_for_pca_ = ca.forward(seed, steps=nca_steps, reading_channel=nca_config['reading_channel'], run_pca=run_PCA, visualise_weights=visualise_weights, visualise_network=visualise_network, inOutdim=[input_dim,action_dim])
            generated_policy_weights = new_pattern.clone().detach()[0]
                
            if run_PCA:
                if len(weights_for_pca) == 0:
                    weights_for_pca = _weights_for_pca_
                else:
                    weights_for_pca = np.concatenate((weights_for_pca, _weights_for_pca_))
                offset = metamorph_times[-1] if len(metamorph_times) != 0 else 0
                metamorph_times.append(nca_steps + offset)
                
            # Load generated weights into policy network 
            reading_channel = nca_config['reading_channel']

            for i in range(nca_config['policy_layers']):
                if i == nca_config['policy_layers'] - 1: # last layer of the policy
                    p.out[2*i].weight = nn.Parameter(generated_policy_weights[reading_channel][i][:action_dim,:], requires_grad=False) 
                else:
                    p.out[2*i].weight = nn.Parameter(generated_policy_weights[reading_channel][i], requires_grad=False) 
            
            if nca_config['NCA_MLP']:
                raise NotImplementedError
                    
            penalty = 0
            if nca_config['penalty_off_topology']:
                raise NotImplementedError
                
            # Prevents runnning environment in case the NCA pattern has died
            if torch.sum(abs(generated_policy_weights)) == 0:
                if render:
                    print('\nThe NCA produce an empty pattern, skipping simulation of the environment.\n')
                return 1000
            if render:
                print(f'\nPolicy {i} sum check is: {torch.sum(abs(generated_policy_weights))}.\n')
                
        
            penalty = 0

            # Burnout phase for the bullet quadruped so it starts off from the floor
            if 'AntBullet' in environment or mujoco_env:
                action = np.zeros(8, dtype=np.float64)
                for _ in range(40):
                    __ = env.step(action)        

            # Inner loop
            neg_count = 0
            rew_ep = 0
            t = 0
            # dim_first_fc = p.out[0].weight.shape[1]
            while True:
                
                # Generate and load policy networks with the NCA
                if nca_config['plastic']:                    
                    raise NotImplementedError
                    
                # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
                if isinstance(env.observation_space, Discrete): 
                    observation = (observation == torch.arange(env.observation_space.n))

                observation = torch.tensor(observation, dtype=torch.float64)
                
                if pixel_env:
                    o3 = p(torch.tensor(observation), dtype=torch.float64)
                else:
                    obs_size = observation.shape[0]
                    o3 = p(observation)
                    # o3 = p(torch.tensor(np.pad(observation, (0, dim_first_fc-obs_size), 'constant', constant_values=0), dtype=torch.float64))[:action_dim]
                    # o3 = p([np.pad(observation, (0, dim_first_fc-observation.shape[0]), 'constant', constant_values=0)])[:action_dim]
                
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
        
                # Environment simulation step
                observation, reward, done, info = env.step(action)  
                if 'Bullet' in environment: reward = env.unwrapped.rewards[1] 
                rew_ep += reward
                
                if 'Bullet' not in environment and render:
                    env.render('human') # Gym envs
                
                if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)
                
                # Early stopping conditions
                if t > 50:
                    neg_count = neg_count+1 if reward < 0.1 else 0
                
                
                if 'Bullet' in environment:
                    if t > 100:
                        if (done or neg_count > 30):
                            break
                else:
                    if done:
                        break
                    if neg_count > 30:
                        break
        
                t += 1
                

            env.close()
            
            rewards.append(rew_ep)
            

    if nca_config['reward_type'] == 'sum':
        cum_reward = (np.sum(rewards) - penalty)
    elif nca_config['reward_type'] == 'min':
        cum_reward = np.min(rewards)
    elif nca_config['reward_type'] == 'mean':
        cum_reward = np.mean(rewards)
    elif nca_config['reward_type'] == 'std':
        cum_reward = (np.sum(rewards) - penalty) - (np.std(np.array(rewards)) / 2)
    else:
        raise NotImplementedError
    
    if debugging or render or visualise_weights or visualise_network:
        print(f"\nEpisode rewards {rewards}\n")
        print(f"\nEpisode {nca_config['reward_type']} reward {cum_reward}\n")
    
    if run_PCA:
        np.save(path + '/' + 'weights_for_PCA.npy', weights_for_pca)
        np.save(path + '/' + 'weights_for_PCA_metamorph_times.npy', np.array(metamorph_times))
        
    return -cum_reward




def evaluate(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='1644785913', metavar='', help='Run id')
    parser.add_argument('--render', type=bool, default=1)
    parser.add_argument('--visualise_weigths', type=bool, default=0)
    parser.add_argument('--visualise_network', type=bool, default=0)
    parser.add_argument('--mean_solution', type=bool, default=1,  help='Whether to use the best population mekan, else it will use best individual solution')
    
    parser.add_argument('--policy_PCA', type=bool, default=0, help= 'Saves weights to run PCA on them')

    args = parser.parse_args()
    
    # Load NCA config
    path = 'saved_models_metamorphosis/'
    nca_config = pickle.load( open( path + args.id + '/nca.config', "rb" ) )   
    
    if 'reward_type' not in nca_config.keys(): # legacy
        nca_config['reward_type'] = 'sum'
    
    for key, value in nca_config.items():
        if key != 'seed':
            print(key,':',value)
    
    # Load evolved NCA weightsl
    for root, dirs, files in os.walk("saved_models_metamorphosis/" + args.id):
        for file in files:
            if args.mean_solution and file and 'meansolution' in file:
                evolved_parameters = np.load(path + args.id + '/' + file) 
                print(f"\nUsing best MEAN solution") 
            elif not args.mean_solution and file and 'bestsolution' in file:
                evolved_parameters = np.load(path + args.id + '/' + file) 
                print(f"\nUsing BEST individual solution") 


    fitnessRL(evolved_parameters=evolved_parameters, nca_config=nca_config,  render=args.render, visualise_weights=args.visualise_weigths, visualise_network=args.visualise_network, run_PCA=args.policy_PCA, path=path+args.id)

    
    
if __name__ == '__main__':
    import argparse
    import torch
    import sys
    evaluate(sys.argv)
