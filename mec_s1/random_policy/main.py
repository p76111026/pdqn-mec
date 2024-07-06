import pickle 
import numpy as np
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
name = "Data/scenarios/scenario_1/global_environment_72604a5e5b364143b36131abaffb8b31.pkl"
env = load_obj(name)
result = []
episode = 4000
for episode_num in range(episode):
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
np.save("random_s1",result)    