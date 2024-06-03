import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from matplotlib import pyplot as plt

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RideHailingEnv:
    def __init__(self, num_vehicles, vehicle_speed, area_size, max_steps):
        self.num_vehicles = num_vehicles
        self.vehicle_speed = vehicle_speed
        self.area_size = area_size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.vehicles = [{
            'position': np.random.uniform(0, self.area_size, 2),
            'state': 'idle',
            'pickup_position': None,
            'dropoff_position': None
        } for _ in range(self.num_vehicles)]
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        state = []
        for vehicle in self.vehicles:
            if vehicle['state'] == 'idle':
                state.extend([0] + vehicle['position'].tolist() + [0, 0, 0, 0])
            else:
                state.extend([1] + vehicle['position'].tolist() + vehicle['pickup_position'].tolist() + vehicle[
                    'dropoff_position'].tolist())
        return np.array(state)

    def step(self, actions):
        rewards = []
        done = False
        for i, action in enumerate(actions):
            reward = 0
            vehicle = self.vehicles[i]
            if vehicle['state'] == 'idle' and action == 1:  # Accept a request
                vehicle['pickup_position'] = np.random.uniform(0, self.area_size, 2)
                vehicle['dropoff_position'] = np.random.uniform(0, self.area_size, 2)
                while np.array_equal(vehicle['pickup_position'], vehicle['dropoff_position']):
                    vehicle['dropoff_position'] = np.random.uniform(0, self.area_size, 2)
                vehicle['state'] = 'occupied'
                reward = 10  # Reward for accepting a ride
            elif vehicle['state'] == 'occupied':
                travel_distance = self.vehicle_speed * (1 / 60)  # Speed in km per minute
                path_vector = vehicle['dropoff_position'] - vehicle['position']
                travel_vector = path_vector / np.linalg.norm(path_vector) * travel_distance
                if np.linalg.norm(travel_vector) > np.linalg.norm(path_vector):
                    vehicle['position'] = vehicle['dropoff_position']
                    reward = 100  # Reward for completing the trip
                    vehicle['state'] = 'idle'
                    vehicle['pickup_position'] = None
                    vehicle['dropoff_position'] = None
                else:
                    vehicle['position'] += travel_vector
                    reward = -1  # Cost per minute while occupied
            rewards.append(reward)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        return self._get_state(), sum(rewards), done


# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# 定义选择动作的函数
def select_action(state, policy_net, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).argmax().item()
    else:
        return random.choice([0, 1])


# 存储经验
def store_experience(memory, state, actions, reward, next_state, done):
    for i, action in enumerate(actions):
        memory.append((state, action, reward, next_state, done))


# 执行经验回放
def experience_replay(memory, policy_net, target_net, optimizer, batch_size, gamma, loss_list):
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
    batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1).to(device)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(device)
    batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
    batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(device)

    current_q_values = policy_net(batch_state).gather(1, batch_action)
    next_q_values = target_net(batch_next_state).max(1)[0].unsqueeze(1).detach()
    expected_q_values = batch_reward + (gamma * next_q_values * (1 - batch_done))

    loss = nn.MSELoss()(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())


# 设置参数
num_vehicles = 10
vehicle_speed = 10  # km/h
area_size = 20  # km
max_steps = 24 * 60
num_episodes = 1000
memory_size = 10000
batch_size = 32
gamma = 0.99
epsilon = 0.95
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
patience = 50  # 提前停止的耐心值

# 初始化环境和DQN
env = RideHailingEnv(num_vehicles, vehicle_speed, area_size, max_steps)
state_size = env.reset().shape[0]
action_size = 2  # 'wait' or 'accept'

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

best_reward = -float('inf')
best_episode = 0
early_stop_counter = 0

# 记录loss和reward
loss_list = []
reward_list = []

# 训练DQN
for episode in range(num_episodes):
    state = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    total_reward = 0
    for step in range(max_steps):
        actions = [select_action(state_tensor, policy_net, epsilon) for _ in range(num_vehicles)]
        next_state, reward, done = env.step(actions)
        total_reward += reward
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
        store_experience(memory, state_tensor.cpu().numpy(), actions, reward, next_state_tensor.cpu().numpy(), done)
        state_tensor = next_state_tensor
        experience_replay(memory, policy_net, target_net, optimizer, batch_size, gamma, loss_list)
        if done:
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if total_reward > best_reward:
        best_reward = total_reward
        best_episode = episode
        early_stop_counter = 0
        torch.save(policy_net.state_dict(), "best_policy_net.pth")  # 保存最优模型
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f"Early stopping at episode {episode} with best reward {best_reward}")
        break

    reward_list.append(total_reward)

    if episode % 1 == 0:
        print(
            f"Episode: {episode}, Total Reward: {total_reward}, Best Reward: {best_reward} at Episode: {best_episode}")

print("Training finished.")

# 绘制reward和loss曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(reward_list)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(loss_list)
plt.title('Loss per Update Step')
plt.xlabel('Update Step')
plt.ylabel('Loss')

plt.show()
