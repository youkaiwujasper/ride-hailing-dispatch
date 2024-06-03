import numpy as np
from scipy.optimize import linear_sum_assignment

# 假设一些初始化数据
vehicle_num = 300
trip_num = 10000
batch_time = 1  # 5 分钟
all_time = 60

# 随机生成车辆初始位置和状态（0 表示空载，1 表示载客）
np.random.seed(0)
vehicle_positions = np.random.randint(0, 100, vehicle_num)
vehicle_status = np.zeros(vehicle_num, dtype=int)  # 初始状态全为空载

# 随机生成订单信息：开始时间、开始位置和目标位置
order_start_times = np.random.randint(0, all_time, trip_num)
print(order_start_times)
order_start_positions = np.random.randint(0, 100, trip_num)
print(order_start_positions)
order_target_positions = np.random.randint(0, 100, trip_num)
print(order_target_positions)


# 构建时间矩阵：位置之间的绝对值差值
def calculate_time_matrix(orders, vehicles):
    return np.abs(np.subtract.outer(orders, vehicles))


# 匹配过程
def hungarian_algorithm(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


# 记录每个 batch_time 的调度情况
dispatch_record = []

# 统计每辆车被分派的订单数量
vehicle_orders_count = np.zeros(vehicle_num, dtype=int)

# 每个 batch_time 进行调度
for t in range(0, all_time, batch_time):  # 假设总时间为 60 分钟
    # 收集当前 batch_time 内的订单
    current_orders = np.where((order_start_times >= t) & (order_start_times < t + batch_time))[0]
    current_order_positions = order_start_positions[current_orders]

    # 获取当前空载的车辆
    empty_vehicles = np.where(vehicle_status == 0)[0]
    empty_vehicle_positions = vehicle_positions[empty_vehicles]

    # 构建时间矩阵
    if len(current_orders) == 0 or len(empty_vehicles) == 0:
        continue

    current_time_matrix = calculate_time_matrix(current_order_positions, empty_vehicle_positions)

    # 使用匈牙利算法进行最优匹配
    order_indices, vehicle_indices = hungarian_algorithm(current_time_matrix)

    # 更新车辆状态和位置
    for oi, vi in zip(order_indices, vehicle_indices):
        order_idx = current_orders[oi]
        vehicle_idx = empty_vehicles[vi]
        vehicle_status[vehicle_idx] = 1  # 变为载客状态
        vehicle_positions[vehicle_idx] = order_target_positions[order_idx]  # 更新车辆位置
        vehicle_orders_count[vehicle_idx] += 1  # 统计车辆被分派的订单数量

    # 记录调度结果
    dispatch_record.append({
        "time": t,
        "orders": current_orders[order_indices],
        "vehicles": empty_vehicles[vehicle_indices],
        "wait_times": current_time_matrix[order_indices, vehicle_indices]
    })

    # 更新车辆状态为空载（假设在本次batch时间内车辆完成服务）
    for vehicle_idx in empty_vehicles[vehicle_indices]:
        vehicle_status[vehicle_idx] = 0  # 订单完成后，车辆变为空载状态

# 打印调度结果
for record in dispatch_record:
    print(f"Batch Time: {record['time']} min")
    for order, vehicle, wait_time in zip(record['orders'], record['vehicles'], record['wait_times']):
        print(f"Order {order} is assigned to Vehicle {vehicle} with waiting time {wait_time} min")

# 计算每辆车的接客总时间和每个订单的等待时间
total_pickup_times = np.zeros(vehicle_num)
total_waiting_times = np.zeros(trip_num)

for record in dispatch_record:
    for vehicle, wait_time in zip(record['vehicles'], record['wait_times']):
        total_pickup_times[vehicle] += wait_time

# 计算每辆车的等待总时间
for record in dispatch_record:
    i = 0
    for order in record['orders']:
        total_waiting_times[order] = record['wait_times'][i]
        i += 1

# 打印每辆车被分派的订单数量
for vehicle_idx, order_count in enumerate(vehicle_orders_count):
    print(f"Vehicle {vehicle_idx} assigned {order_count} orders")

print("Total pickup times for each vehicle:", total_pickup_times)
print("Waiting times for each order:", total_waiting_times)
