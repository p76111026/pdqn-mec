import os
import click
import time
from MPDQN_mec.common import ClickPythonLiteralOption
import numpy as np
import pickle
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem, christofides
import itertools
import multiprocessing as mp
def load_obj(name):
    """
    Loads a pickle file object
    :param name:
    :return:
    """
    with open(name, 'rb') as f:
        return pickle.load(f)

def offloading_decision(decimal_number,tsp_path,edge_index):
    ternary_number = ""
    if decimal_number == 0:
        now_index = np.where(tsp_path == edge_index)[0]
        now_index -= 1
        return [tsp_path[(now_index)%9][0],tsp_path[(now_index)%9][0],tsp_path[(now_index)%9][0]]
    else:
        while decimal_number > 0:
            remainder = decimal_number % 3
            ternary_number = str(remainder) + ternary_number
            decimal_number = decimal_number // 3

    strategy = [int(ternary_number)//100,int(ternary_number)%100//10,int(ternary_number)%10]
    now_index = np.where(tsp_path == edge_index)[0]
    now_index -= 1
    return [tsp_path[(now_index+strategy[0])%9][0],tsp_path[(now_index+strategy[1])%9][0],tsp_path[(now_index+strategy[2])%9][0]]
# def distance(edge_xs,edge_ys,computing_speed,edge_index_1, edge_index_2, beta=2000):
#     part_1 = ((edge_xs[edge_index_1] - edge_xs[edge_index_2])**2 + (edge_ys[edge_index_1] - edge_ys[edge_index_2])**2)**0.5
#     part_2 = max(computing_speed[edge_index_1] / computing_speed[edge_index_2], computing_speed[edge_index_2] / computing_speed[edge_index_1])
#     return part_1 + beta * part_2**-1
def distance(edge_xs,edge_ys,computing_speed,edge_index_1, edge_index_2, beta=1):
    part_1 = ((edge_xs[edge_index_1] - edge_xs[edge_index_2])**2 + (edge_ys[edge_index_1] - edge_ys[edge_index_2])**2)**0.5
    part_2 = min(computing_speed[edge_index_1] / computing_speed[edge_index_2], computing_speed[edge_index_2] / computing_speed[edge_index_1])
    normalize_part_1 = (part_1-1000)/(2000*1.44-1000)
    normalize_part_2 = (part_2-0.3)/(1-0.3)
    return normalize_part_1 + beta*normalize_part_2
def calculate_path(edge_xs,edge_ys,new_computing_speeds,perm):
    dist = 0
    for index in range(8):
        dist += distance(edge_xs, edge_ys, new_computing_speeds, perm[index], perm[index + 1])
    dist += distance(edge_xs, edge_ys, new_computing_speeds, perm[0], perm[-1])   
    return dist, perm
@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=4100, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=256, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.996, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--inverting-gradients', default=True,
            help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.',
            type=int)
@click.option('--use-ornstein-noise', default=False,
            help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=100000, help='Replay memory size in transitions.', type=int) # 500000
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.1, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.001, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=1., help="Gradient clipping.", type=float)  # 1 better than 10.
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)  # 0.5
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters at when using split Q-networks.', type=int)
@click.option('--qlayers', default="[256,128,64]", help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--paramlayers', default="[64,128,256]", help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="result/mp_dqn_tsp_newstate_randomtasksamespeedsoccur_s2/", help='Output directory.', type=str)
@click.option('--title', default="MPDQN", help="Prefix of output files", type=str)
def run(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor, learning_rate_actor_param, title, epsilon_final,
        clip_grad, beta, scale_actions, split, indexed, zero_index_gradients, action_input_layer,
        evaluation_episodes, multipass, weighted, average, random_weighted, update_ratio,
        save_freq, save_dir, qlayers,paramlayers):

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)

    name ="Data/scenarios/scenario_1/global_environment_72604a5e5b364143b36131abaffb8b31.pkl"
    env = load_obj(name)
    dir = os.path.join(save_dir, title)
    
    from MPDQN_mec.agents.pdqn_mec import PDQNAgent
    from MPDQN_mec.agents.pdqn_mec_multipass import MultiPassPDQNAgent
    assert not (split and multipass)
    agent_class = PDQNAgent
    if split:
        pass
    elif multipass:
        agent_class = MultiPassPDQNAgent
    assert action_input_layer >= 0
    if action_input_layer > 0:
        assert split
    agent = agent_class(
                        env.edge_observation_spec(), env.edge_action_spec(),
                        actor_kwargs={"hidden_layers": qlayers,
                                        'action_input_layer': action_input_layer,
                                        'activation': "leaky_relu",
                                        'output_layer_init_std': 0.01},
                        actor_param_kwargs={"hidden_layers": paramlayers,
                                            'activation': "leaky_relu",
                                            'output_layer_init_std': 0.01},
                        batch_size=batch_size,
                        learning_rate_actor=learning_rate_actor,  # 0.0001
                        learning_rate_actor_param=learning_rate_actor_param,  # 0.001
                        epsilon_steps=epsilon_steps,
                        epsilon_final=epsilon_final,
                        gamma=gamma,  # 0.99
                        tau_actor=tau_actor,
                        tau_actor_param=tau_actor_param,
                        clip_grad=clip_grad,
                        #beta=beta,
                        indexed=indexed,
                        weighted=weighted,
                        average=average,
                        random_weighted=random_weighted,
                        initial_memory_threshold=initial_memory_threshold,
                        use_ornstein_noise=use_ornstein_noise,
                        replay_memory_size=replay_memory_size,
                        inverting_gradients=inverting_gradients,
                        zero_index_gradients=zero_index_gradients,
                        seed=seed)
    print(agent)
    network_trainable_parameters = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    network_trainable_parameters += sum(p.numel() for p in agent.actor_param.parameters() if p.requires_grad)
    print("Total Trainable Network Parameters: %d" % network_trainable_parameters)
    
    
    result = {
            'cumulative_reward': [],
            'average_vehicle_SINRs': [],
            'average_vehicle_intar_interference': [],
            'average_vehicle_inter_interference': [],
            'average_vehicle_interferences': [],
            'average_transmision_times': [],
            'average_wired_transmission_times': [],
            'average_execution_times': [],
            'average_service_times': [],
            'service_rate': [],
            'offload_rate': [],
            'local_rate': [],
            'occu_power':[],
            'occu_computing_resource':[],
            'tsp':[],
            "decision":[]
            }
    num_update  = 0 
    edge_xs = [500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500]
    edge_ys = [2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500]
    
    np.random.seed(0)
    
    
    new_computing_speeds = [3.0 * 1e9, 10.0 * 1e9, 3.0 * 1e9, 10.0 * 1e9, 6.0 * 1e9, 10.0 * 1e9, 3.0 * 1e9, 10.0 * 1e9, 3.0 * 1e9]
    #new_computing_speeds = [10.0 * 1e9, 10.0 * 1e9, 10.0 * 1e9, 10.0 * 1e9, 6.0 * 1e9, 3.0 * 1e9, 3.0 * 1e9, 3.0 * 1e9, 3.0 * 1e9]
    #new_computing_speeds = [10.0 * 1e9, 10.0 * 1e9, 10.0 * 1e9, 10.0 * 1e9, 3.0 * 1e9, 3.0 * 1e9, 3.0 * 1e9, 3.0 * 1e9, 6.0 * 1e9]
    
    
    
    tsp_path = [0,1,2,3,4,5,6,7,8]
    tsp_path = np.array(tsp_path)
    for episode_num in range(episodes):
        start = time.time()
        for i in range(9):
            env._edge_list._edge_list[i]._computing_speed = new_computing_speeds[i]
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i)))
        info = {'status': "NOT_SET"}
        timestep = env.reset()
        state = np.array(timestep.observation, dtype=np.float32, copy=False)
        state = state.reshape(-1,26)
        #for i in range(9):
        #    state[i][15:24] = new_computing_speeds/1e9
        act, act_param, all_action_parameters = agent.act(state)
        act_param = np.array(act_param)
        
        step = 0
        agent.start_episode()

        cumulative_rewards: float = 0
        average_vehicle_SINRs: float = 0
        average_vehicle_intar_interferences: float = 0
        average_vehicle_inter_interferences: float = 0
        average_vehicle_interferences: float = 0
        average_transmision_times: float = 0
        average_wired_transmission_times: float = 0
        average_execution_times: float = 0
        average_service_times: float = 0 
        successful_serviced_numbers: float = 0
        task_required_numbers: float = 0
        task_offloaded_numbers: float = 0
        average_service_rate: float = 0
        average_offloaded_rate: float = 0
        average_local_rate: float = 0
        
        while not timestep.last():
            step += 1
            action = np.zeros(243)
            for i in range(9):
                action[27*i:27+27*i] = np.insert(act_param[i], 3, offloading_decision(act[i],tsp_path,i))
            #next_state, reward, terminal, info = env.step(action)
            timestep, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
                average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number= env.step(action)
            cumulative_rewards = cumulative_rewards + cumulative_reward   
            average_vehicle_SINRs += average_vehicle_SINR
            average_vehicle_intar_interferences += average_vehicle_intar_interference
            average_vehicle_inter_interferences += average_vehicle_inter_interference 
            average_vehicle_interferences += average_vehicle_interference
            average_transmision_times += average_transmision_time
            average_wired_transmission_times += average_wired_transmission_time
            average_execution_times += average_execution_time
            average_service_times += average_service_time
            successful_serviced_numbers += successful_serviced_number
            task_required_numbers += task_required_number
            task_offloaded_numbers += task_offloaded_number
            

            
            
            next_state = np.array(timestep.observation, dtype=np.float32, copy=False)
            next_state = next_state.reshape(-1, 26)
            #for i in range(9):
            #    next_state[i][15:24] = new_computing_speeds/1e9
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_act_param = np.array(next_act_param)
            
            for i in range(9):
                agent.replay_memory.append(state=state[i], action=np.concatenate(([act[i]], all_action_parameters[i].data)).ravel(), reward=timestep.reward[i], next_state=next_state[i], next_action=np.concatenate(([next_act[i]], next_all_action_parameters[i].data)).ravel(),
                                    terminal=timestep.last())
                agent._step += 1 
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            result["decision"].append(act)
            state = next_state
        
            
            if timestep.last():
                break
        if episode_num <= 4000 :
            agent._optimize_td_loss()
        #env.render()
        average_vehicle_SINRs /= task_required_numbers
        average_vehicle_intar_interferences /= task_required_numbers
        average_vehicle_inter_interferences /= task_required_numbers 
        average_vehicle_interferences /= task_required_numbers
        
        average_transmision_times /= task_required_numbers
        average_wired_transmission_times /= task_required_numbers
        average_execution_times /= task_required_numbers
        average_service_times /= task_required_numbers
        
        
        average_service_rate = successful_serviced_numbers / task_required_numbers
        average_offloaded_rate = task_offloaded_numbers / task_required_numbers
        average_local_rate = (task_required_numbers - task_offloaded_numbers) / task_required_numbers
            
        
        
        result['cumulative_reward'].append( cumulative_rewards)
        result['average_vehicle_SINRs'].append(average_vehicle_SINRs)
        result['average_vehicle_intar_interference'].append(average_vehicle_intar_interferences)
        result['average_vehicle_inter_interference'].append(average_vehicle_inter_interferences)
        result['average_vehicle_interferences'].append(average_vehicle_interferences)
        result['average_transmision_times'].append(average_transmision_times)
        result['average_wired_transmission_times'].append(average_wired_transmission_times)
        result['average_execution_times'].append(average_execution_times)
        result['average_service_times'].append(average_service_times)
        result['service_rate'].append(average_service_rate)
        result['offload_rate'].append(average_offloaded_rate)
        result['local_rate'].append(average_local_rate)    
        result['tsp'].append(tsp_path)
        
        agent.end_episode()
        print(f"episode: {episode_num+1}   reward: {cumulative_rewards} time: {time.time()-start} ")
        
    np.save("mp_dqn_s1",result)
    #agent.save_models(os.path.join(save_dir))
    #if save_freq > 0 and save_dir:
    #    agent.save_models(os.path.join(save_dir, str(i)))



if __name__ == '__main__':
    run()
