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
mpdqn = np.load("mp_dqn_s1_10e5.npy", allow_pickle=True)
mpdqn_tsp = np.load("mp-dqn-tsp_s1_10e5.npy", allow_pickle=True)
random = np.load("random_s1.npy", allow_pickle=True)
mad4pg = np.load("mad4pg.npy", allow_pickle=True)

data = mpdqn.item()['cumulative_reward']
data_2 = mpdqn_tsp.item()['cumulative_reward']

# 绘制图像
fig, ax = plt.subplots(figsize=(10, 10))
draw("mad4pg.npy", 'cumulative_reward')
ax.plot(moving_average(data[:4000], 50), color="green", label='MP-DQN')
ax.plot(moving_average(data_2[:4000], 50), color='blue', label='MP-DQN-TSP')
ax.plot(moving_average(random[:4000], 50), color='purple', label='Random')

# 添加水平的标签
ax.set_xlabel('episode')
ax.set_ylabel('cumulative reward')

# 添加图例
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

# 保存图像时指定 dpi
fig.savefig('my_figure_200dpi.png', dpi=200)

# 显示图像
plt.show()
