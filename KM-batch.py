import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import h5py

# 读取清洗后的行程数据
df = pd.read_csv("./Data/20150401_cleaned.csv")

# 加载行程时间数据
trip_times_file = h5py.File('trip_times.h5', 'r')

# 假设一些初始化数据
vehicle_num = 300
batch_time = 5  # 1 分钟
all_time = 24 * 60  # 24小时，也就是24*60分钟

# 定义经纬度范围
min_longitude, max_longitude = 103.94, 104.21
min_latitude, max_latitude = 30.578, 30.788

# 生成车辆位置的节点
vehicle_positions = np.random.choice(df['start_node'].unique(), vehicle_num)

# 随机生成车辆状态（0 表示空载，1 表示载客）
vehicle_status = np.zeros(vehicle_num, dtype=int)  # 初始状态全为空载

# 提取订单信息并转换时间为分钟
df['GETONTIME'] = pd.to_datetime(df['GETONTIME'])
df['GETOFFTIME'] = pd.to_datetime(df['GETOFFTIME'])
df['GETONTIME_MIN'] = (df['GETONTIME'] - df['GETONTIME'].dt.floor('d')).dt.total_seconds() // 60
df['GETOFFTIME_MIN'] = (df['GETOFFTIME'] - df['GETOFFTIME'].dt.floor('d')).dt.total_seconds() // 60

order_start_times = df['GETONTIME_MIN'].values
order_end_times = df['GETOFFTIME_MIN'].values
order_start_nodes = df['start_node'].values
order_end_nodes = df['end_node'].values


# 读取行程时间的函数
def get_trip_time(start_node, end_node, current_time):
    hour = current_time // 60  # 从分钟数转换为小时数
    return trip_times_file[f'hour_{hour}'][start_node, end_node]


# 构建时间矩阵
def calculate_time_matrix(orders, vehicles, current_time):
    time_matrix = np.zeros((len(orders), len(vehicles)))
    for i, start_node in enumerate(orders):
        for j, vehicle_node in enumerate(vehicles):
            time_matrix[i, j] = get_trip_time(vehicle_node, start_node, current_time)
    return time_matrix


def hungarian_algorithm(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


dispatch_record = []
vehicle_orders_count = np.zeros(vehicle_num, dtype=int)

for t in range(0, all_time, batch_time):  # 假设总时间为 24 小时
    for idx, record in enumerate(dispatch_record):
        completed_vehicles = np.where(np.array(record['completion_times']) <= t)[0]
        for cv in completed_vehicles:
            vehicle_status[record['vehicles'][cv]] = 0

    current_orders = np.where((order_start_times >= t) & (order_start_times < t + batch_time))[0]
    current_order_nodes = order_start_nodes[current_orders]

    empty_vehicles = np.where(vehicle_status == 0)[0]
    empty_vehicle_nodes = vehicle_positions[empty_vehicles]

    if len(current_orders) == 0 or len(empty_vehicles) == 0:
        continue

    current_time_matrix = calculate_time_matrix(current_order_nodes, empty_vehicle_nodes, t)

    order_indices, vehicle_indices = hungarian_algorithm(current_time_matrix)

    pickup_times = []
    completion_times = []
    for oi, vi in zip(order_indices, vehicle_indices):
        order_idx = current_orders[oi]
        vehicle_idx = empty_vehicles[vi]
        vehicle_status[vehicle_idx] = 1
        pickup_time = t + current_time_matrix[oi, vi]
        completion_time = pickup_time + (order_end_times[order_idx] - order_start_times[order_idx])
        pickup_times.append(pickup_time)
        completion_times.append(completion_time)
        vehicle_positions[vehicle_idx] = order_end_nodes[order_idx]
        vehicle_orders_count[vehicle_idx] += 1

    dispatch_record.append({
        "time": t,
        "orders": current_orders[order_indices],
        "vehicles": empty_vehicles[vehicle_indices],
        "pickup_times": pickup_times,
        "completion_times": completion_times
    })

for record in dispatch_record:
    print(f"Batch Time: {record['time']} min")
    for order, vehicle, pickup_time, completion_time in zip(record['orders'], record['vehicles'],
                                                            record['pickup_times'], record['completion_times']):
        print(
            f"Order {order} is assigned to Vehicle {vehicle} with pickup time {pickup_time} min and completion time {completion_time} min")

total_pickup_times = np.zeros(vehicle_num)
total_waiting_times = np.zeros(len(df))

for record in dispatch_record:
    for vehicle, pickup_time, completion_time in zip(record['vehicles'], record['pickup_times'],
                                                     record['completion_times']):
        total_pickup_times[vehicle] += (completion_time - pickup_time)

for record in dispatch_record:
    for i, order in enumerate(record['orders']):
        total_waiting_times[order] = record['pickup_times'][i] - order_start_times[order]

for vehicle_idx, order_count in enumerate(vehicle_orders_count):
    print(f"Vehicle {vehicle_idx} assigned {order_count} orders")

print("Total pickup times for each vehicle:", total_pickup_times)
print("Waiting times for each order:", total_waiting_times)
