import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """
    使用 numpy 提供的函數計算移動平均值。
    
    參數:
    data -- 輸入的NumPy數組
    window_size -- 窗口的大小
    
    返回:
    NumPy數組，其中包含每個窗口的平均值
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def draw(name, content):
    test = np.load(name, allow_pickle=True)
    temp = [i[content] for i in test]
    plt.plot(moving_average(temp, 50), color='red', label='MAD4PG')

import torch
import pandas as pd
import seaborn as sns

# 加载数据
mpdqn = np.load("mp_dqn_s2.npy", allow_pickle=True)
mpdqn_tsp = np.load("mp_dqn_tsp_s2.npy", allow_pickle=True)
random = np.load("random_s2.npy", allow_pickle=True)
mad4pg = np.load("mad4pg.npy", allow_pickle=True)


offload_rate = mpdqn.item()['offload_rate']
offload_rate_2 = mpdqn_tsp.item()['offload_rate']
test = np.load("mad4pg.npy",allow_pickle=True)
temp = []
for i in test:
    temp.append(i["offload_rate"])
print(sum(temp[4000:])/100)
data = [sum(offload_rate[4000:])/100,sum(offload_rate_2[4000:])/100,sum(temp[4000:])/100]
algorithm_names=["mp-dqn","mp-dqn-tsp","mad4pg"]
plt.bar(algorithm_names, data, color=['green', 'blue', 'red'])
plt.xlabel('Algorithms')
plt.ylabel('offload ratio')
plt.savefig("offload_ratio_comparison_dpi200.png", dpi=200)
plt.show()


local_rate = mpdqn.item()['local_rate']
local_rate_2 = mpdqn_tsp.item()['local_rate']
test = np.load("mad4pg.npy",allow_pickle=True)
temp = []
for i in test:
    temp.append(i["local_rate"])
print(sum(temp[4000:])/100)
data = [sum(local_rate[4000:])/100,sum(local_rate_2[4000:])/100,sum(temp[4000:])/100]
algorithm_names=["mp-dqn","mp-dqn-tsp","mad4pg"]
plt.bar(algorithm_names, data, color=['green', 'blue', 'red'])
plt.xlabel('Algorithms')
plt.ylabel('local ratio')
plt.savefig("local_ratio_comparison_dpi200.png", dpi=200)
plt.show()

service_rate = mpdqn.item()['service_rate']
service_rate_2 = mpdqn_tsp.item()['service_rate']
test = np.load("mad4pg.npy",allow_pickle=True)
temp = []
for i in test:
    temp.append(i["service_rate"])
print(sum(temp[4000:])/100)
data = [sum(service_rate[4000:])/100,sum(service_rate_2[4000:])/100,sum(temp[4000:])/100]
algorithm_names=["mp-dqn","mp-dqn-tsp","mad4pg"]
plt.bar(algorithm_names, data, color=['green', 'blue', 'red'])
plt.xlabel('Algorithms')
plt.ylabel('service ratio')
plt.savefig("service_ratio_comparison_dpi200.png", dpi=200)
plt.show()



service_time = mpdqn.item()['average_service_times']
service_time_2 = mpdqn_tsp.item()['average_service_times']
test = np.load("mad4pg.npy",allow_pickle=True)
temp = []
for i in test:
    temp.append(i["average_service_times"])
print(sum(temp[4000:])/100)
data = [sum(service_time[4000:])/100,sum(service_time_2[4000:])/100,sum(temp[4000:])/100]
algorithm_names=["mp-dqn","mp-dqn-tsp","mad4pg"]
plt.bar(algorithm_names, data, color=['green', 'blue', 'red'])
plt.xlabel('Algorithms')
plt.ylabel('average service time')
plt.savefig("service_time_comparison_dpi200.png", dpi=200)
plt.show()

reward = mpdqn.item()['cumulative_reward']
reward_2 = mpdqn_tsp.item()['cumulative_reward']
test = np.load("mad4pg.npy",allow_pickle=True)
temp = []
for i in test:
    temp.append(i["cumulative_reward"])
print(sum(temp[4000:])/100)
data = [sum(reward[4000:])/100,sum(reward_2[4000:])/100,sum(temp[4000:])/100]
algorithm_names=["mp-dqn","mp-dqn-tsp","mad4pg"]
plt.bar(algorithm_names, data, color=['green', 'blue', 'red'])
plt.xlabel('Algorithms')
plt.ylabel('cumulative_reward')
plt.savefig("reward_comparison_dpi200.png", dpi=200)
plt.show()

print(sum(reward[4000:])/100,sum(reward_2[4000:])/100)
print("----------")
reward = mpdqn.item()['cumulative_reward']
reward_2 = mpdqn_tsp.item()['cumulative_reward']
test = np.load("mad4pg.npy",allow_pickle=True)
temp = []
for i in test:
    temp.append(i["cumulative_reward"])
#print(sum(temp[4000:])/100)
data = [reward[4000:],reward_2[4000:],temp[4000:]]
algorithm_names=["mp-dqn","mp-dqn-tsp","mad4pg"]
plt.boxplot(data, patch_artist=True)
plt.xticks([1,2,3],algorithm_names )
plt.xlabel('Algorithms')
plt.ylabel('cumulative_reward')
plt.savefig("reward_comparison_box_dpi200.png", dpi=200)
plt.show()
