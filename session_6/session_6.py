import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode='rgb_array')

# Detect whether CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x

# Use double-ended queue to store memory
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Return a random batch of sample in the memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

dqn = DQN(state_size, action_size).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
memory = ReplayBuffer(10000)

batch_size = 64
gamma = 0.99  # discount factor
epsilon = 1.0

num_episodes = 500

filtered_reward = 0

for episode in range(num_episodes):
    # reset() returns the initial observation
    state = env.reset()[0]
    done = False;
    total_reward = 0

    while not done:
        # Epsilon greedy policy
        # Sometimes choose action randomly, sometimes choose according to the model
        # The probability of choosing randomly will decay over time
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            q_values = dqn(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done, _, _ = env.step(action)

        # Push the current step into memory buffer
        memory.add(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(memory.memory) >= batch_size:
            transitions = memory.sample(batch_size)
            batch = np.array(transitions, dtype=object).transpose()
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

            state_batch = torch.tensor(np.stack(state_batch), device=device, dtype=torch.float32)
            action_batch = torch.tensor(np.array(action_batch, dtype=np.int64), device=device, dtype = torch.int64)
            reward_batch = torch.tensor(np.array(reward_batch, dtype=np.float32), device=device, dtype=torch.float32)
            next_state_batch = torch.tensor(np.stack(next_state_batch), device=device, dtype=torch.float32)
            not_done_mask = torch.tensor(~np.array(done_batch, dtype=np.bool_), device=device, dtype=torch.bool)

            current_q_values = dqn(state_batch).gather(1, action_batch.unsqueeze(1))
            next_max_q_values = dqn(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (gamma * next_max_q_values * not_done_mask)

            loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(dqn.parameters(), 100)
            optimizer.step()

    # Epsilon decay
    if epsilon > 0.05:
        epsilon *= 0.995

    filtered_reward = 0.9 * filtered_reward + 0.1 * total_reward

    if episode % 50 == 0:
        print(f'Episode {episode}/{num_episodes}, Filtered Total Reward: {filtered_reward}')

import matplotlib.animation as animation

fig, _ = plt.subplots()

def test_agent(env, trained_agent):
    frames = []
    state = env.reset()[0]
    done = False
    while not done and len(frames) < 1000:  # It may take too much time to render until termination
        frames.append(env.render())

        with torch.no_grad():
            q_values = trained_agent(torch.tensor(np.array(state), device=device, dtype=torch.float32).unsqueeze(0))
            action = torch.argmax(q_values).item()

        state, _, done, _, _ = env.step(action)

    env.close()
    return frames

def animate_frames(frames):
    img = plt.imshow(frames[0])

    def update(frame):
        img.set_array(frame)
        return img,

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)
    plt.axis('off')
    plt.show()

trained_agent = dqn
frames = test_agent(env, trained_agent)
animate_frames(frames)
