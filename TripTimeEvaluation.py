import numpy as np
import pandas as pd
import networkx as nx
import joblib  # 用于加载模型
import h5py

# 读取路网信息
road_network = pd.read_csv("./Data/chengdu-link.csv")

# 构建图数据结构，并添加纬度和经度信息
G = nx.DiGraph()
for _, row in road_network.iterrows():
    G.add_node(row['Node_Start'], Latitude=row['Latitude_Start'], Longitude=row['Longitude_Start'])
    G.add_node(row['Node_End'], Latitude=row['Latitude_End'], Longitude=row['Longitude_End'])
    G.add_edge(row['Node_Start'], row['Node_End'], weight=row['Length'])

# 加载训练好的随机森林模型
model = joblib.load('random_forest_model.pkl')

# 获取所有节点
nodes = list(G.nodes())


# 估算行程时间的函数
def estimate_travel_time(start_node, end_node, hour):
    # 计算最短路径长度
    try:
        path_length = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')
    except nx.NetworkXNoPath:
        path_length = np.inf  # 如果没有路径，则设为无穷大

    # 构建特征向量
    features = pd.DataFrame({
        'path_length': [path_length],
        'start_hour': [hour],
        'start_lat': [G.nodes[start_node]['Latitude']],
        'start_lon': [G.nodes[start_node]['Longitude']],
        'end_lat': [G.nodes[end_node]['Latitude']],
        'end_lon': [G.nodes[end_node]['Longitude']]
    })

    # 估算行程时间
    estimated_time = model.predict(features)[0]
    return estimated_time


# 计算并存储每个小时的行程时间
def calculate_and_store_trip_times(hour):
    print(f"Calculating trip times for hour {hour}")
    trip_times = np.zeros((len(nodes), len(nodes)))
    for i, start_node in enumerate(nodes):
        print(f"calculating node {i}")
        for j, end_node in enumerate(nodes):
            trip_times[i, j] = estimate_travel_time(start_node, end_node, hour)

    with h5py.File(f'trip_times_hour_{hour}.h5', 'w') as f:
        f.create_dataset(f'hour_{hour}', data=trip_times)


for hour in range(1, 2):
    calculate_and_store_trip_times(hour)

print("行程时间计算并存储完成。")
