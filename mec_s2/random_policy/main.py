import pickle 
import numpy as np
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
name = "Data/scenarios/scenario_1/global_environment_72604a5e5b364143b36131abaffb8b31.pkl"
env = load_obj(name)
env._occuiped=True
env._config.task_minimum_data_size = env._config.task_minimum_data_size*0.3
env._config.task_maximum_data_size = env._config.task_maximum_data_size*0.3
env._config.task_minimum_delay_thresholds =  env._config.task_minimum_delay_thresholds*0.3
env._config.task_minimum_delay_thresholds =  env._config.task_maximum_delay_thresholds*0.3
new_computing_speeds = [10.0 * 1e9, 10.0 * 1e9, 10.0 * 1e9, 10.0 * 1e9, 6.0 * 1e9, 3.0 * 1e9, 3.0 * 1e9, 3.0 * 1e9, 3.0 * 1e9]
result = []
episode = 4000
for episode_num in range(episode):
    env._config.task_seed = episode_num
    for i in range(9):
            env._edge_list._edge_list[i]._computing_speed = new_computing_speeds[i]
    timestep = env.reset()
    cumulative_rewards = 0
    while not timestep.last():
        np.random.seed(episode_num)
        action = np.random.rand(243)
        timestep, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
                average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number = env.step(action)
        cumulative_rewards = cumulative_rewards + cumulative_reward
    result.append(cumulative_rewards)
    print(f"{episode_num}:reward:{cumulative_rewards}")
np.save("random_s2",result)    