#####################
#這段程式碼是用來繪製每個edge在300個step當中所需要服務的車輛之路徑圖
# 對應論文 fig 8 車輛軌跡
#####################
import pickle 
import matplotlib.pyplot as plt

def load_obj(name):
    """
    Loads a pickle file object
    :param name:
    :return:
    """
    with open(name, 'rb') as f:
        return pickle.load(f)

name = "Data/scenarios/scenario_1/global_environment_72604a5e5b364143b36131abaffb8b31.pkl"
env = load_obj(name)

# 定義軌跡的顏色
trajectory_colors = ['r', 'g', 'b']

for i in range(27):
    trajectory = env._vehicle_list.get_vehicle_by_index(vehicle_index=i).get_vehicle_trajectory()

    trajectory_data_str = str(trajectory).replace('\n', '')

    # 解析轨迹数据字符串
    trajectory_data = [tuple(map(float, item.strip('()').split(','))) for item in trajectory_data_str.split(')(')]

    # 提取经度和纬度
    longitudes = [data[1] for data in trajectory_data]
    latitudes = [data[2] for data in trajectory_data]

    # 獲取車輛的顏色索引
    color_index = i % 3

    # 绘制路径图，根據顏色索引選擇顏色
    plt.plot(longitudes, latitudes, marker='o', linestyle='-', color=trajectory_colors[color_index], label=f'Trajectory {i+1}' if i < 3 else "")

# 繪製邊上的車輛標記
edge_xs = [500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500]
edge_ys = [2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500]

for i in range(9):
    plt.scatter(edge_xs[i], edge_ys[i], marker='^', color='r', label='Server' if i == 0 else "") 

    # 繪製通訊範圍圓
    circle = plt.Circle((edge_xs[i], edge_ys[i]), 500, color='r', fill=False, linestyle='--', label='Communication Range' if i == 0 else "")
    plt.gca().add_patch(circle)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Vehicle Trajectory')
plt.grid(True)
plt.legend()  # 添加圖例
plt.show()
