import pandas as pd
from sklearn.neighbors import BallTree
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# 读取路网数据
road_data = pd.read_csv('./data/chengdu-link.csv', encoding='GBK')
# 读取行程数据
trip_data = pd.read_csv('./data/20150401-0403-trip2link.csv', encoding='GBK')

# 提取路网节点
road_points = []

for _, row in road_data.iterrows():
    road_points.append({'lat': row['Latitude_Start'], 'lon': row['Longitude_Start'], 'point': row['Node_Start']})
    road_points.append({'lat': row['Latitude_End'], 'lon': row['Longitude_End'], 'point': row['Node_End']})

# 去重
road_points = [dict(t) for t in {tuple(d.items()) for d in road_points}]

# 创建BallTree
points = np.array([(point['lat'], point['lon']) for point in road_points])
tree = BallTree(points, leaf_size=40)


# 创建一个查找点到最近节点的函数
def find_nearest_node(lat, lon):
    dist, idx = tree.query(np.array([[lat, lon]]), k=1)
    return road_points[idx[0][0]]['point']


# 构建路网图
G = nx.DiGraph()

for _, row in road_data.iterrows():
    G.add_edge(row['Node_Start'], row['Node_End'], length=row['Length'])

# 构建特征和标签
features = []
labels = []
trip_data['start_node'] = None
trip_data['end_node'] = None

for i, trip in trip_data.iterrows():
    if i % 10000 == 0:
        print(f"{i} trips matched.")
    start_node = find_nearest_node(trip['GETONLATITUDE'], trip['GETONLONGITUDE'])
    end_node = find_nearest_node(trip['GETOFFLATITUDE'], trip['GETOFFLONGITUDE'])
    path_length = nx.shortest_path_length(G, source=start_node, target=end_node, weight='length')

    start_time = pd.to_datetime(trip['GETONTIME'])
    end_time = pd.to_datetime(trip['GETOFFTIME'])

    features.append([
        path_length,
        start_time.hour,
        trip['GETONLATITUDE'],
        trip['GETONLONGITUDE'],
        trip['GETOFFLATITUDE'],
        trip['GETOFFLONGITUDE']
    ])
    labels.append((end_time - start_time).total_seconds() / 60)  # 转换为分钟

    trip_data.at[i, 'start_node'] = start_node
    trip_data.at[i, 'end_node'] = end_node

# 保存含有匹配节点的行程数据到新的CSV文件
trip_data.to_csv('20150401-0403-trip2node.csv', index=False)

# 转换为DataFrame
df = pd.DataFrame(features, columns=['path_length', 'start_hour', 'start_lat', 'start_lon', 'end_lat', 'end_lon'])
df['trip_time'] = labels

# 使用KMeans聚类进行离群值检测
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['path_length', 'trip_time']])

kmeans = KMeans(n_clusters=5, random_state=42).fit(scaled_features)
df['cluster'] = kmeans.labels_

# 计算每个点到其聚类中心的距离
df['distance_to_center'] = np.linalg.norm(scaled_features - kmeans.cluster_centers_[kmeans.labels_], axis=1)

# 选择阈值，将距离较大的点视为离群值
threshold = df['distance_to_center'].quantile(0.95)
df_filtered = df[df['distance_to_center'] <= threshold]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    df_filtered[['path_length', 'start_hour', 'start_lat', 'start_lon', 'end_lat', 'end_lon']],
    df_filtered['trip_time'], test_size=0.2, random_state=42)

# 训练模型
model = RandomForestRegressor()
model.fit(X_train, y_train)


# 预测函数
def predict_trip_time(start_lat, start_lon, end_lat, end_lon, start_time):
    start_node = find_nearest_node(start_lat, start_lon)
    end_node = find_nearest_node(end_lat, end_lon)
    path_length = nx.shortest_path_length(G, source=start_node, target=end_node, weight='length')

    features = [[path_length, start_time.hour, start_lat, start_lon, end_lat, end_lon]]
    predicted_time = model.predict(features)

    return predicted_time


# 测试部分
y_pred = model.predict(X_test)

# 计算评估指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')

# 打印部分预测结果与实际结果
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

# 保存模型
joblib.dump(model, 'random_forest_model.pkl')

'''
Mean Absolute Error (MAE): 2.625599280949477
Mean Squared Error (MSE): 19.379473179279135
Root Mean Squared Error (RMSE): 4.4022123051119575
Mean Absolute Percentage Error (MAPE): 0.2695417862358931
'''
