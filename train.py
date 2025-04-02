import numpy as np
import random
from env import DroneGridEnv
import time

class QLearningAgent:
    def __init__(self):
        # Q-table: 8x8x8x8 grid × 4 actions
        self.q_table = np.zeros((8, 8, 8, 8, 4))
        
        # Hyperparameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Tracking
        self.visited = np.zeros((8, 8))  # Track visited states

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        drone_x, drone_y, target_x, target_y = state
        
        # Update visit count
        self.visited[drone_x, drone_y] += 1
        
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Random action
        return np.argmax(self.q_table[drone_x, drone_y, target_x, target_y])

    def update_q_table(self, state, action, reward, next_state):
        """Q-learning update rule"""
        drone_x, drone_y, target_x, target_y = state
        next_drone_x, next_drone_y, _, _ = next_state
        
        current_q = self.q_table[drone_x, drone_y, target_x, target_y, action]
        max_next_q = np.max(self.q_table[next_drone_x, next_drone_y, target_x, target_y])
        
        # Q-learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[drone_x, drone_y, target_x, target_y, action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_q_learning(episodes=1000):
    env = DroneGridEnv()
    agent = QLearningAgent()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get action and execute
            action = agent.get_action(state)
            direction = ['up', 'down', 'left', 'right'][action]
            moved = env.move(direction)
            
            # Get reward and new state
            reward = env.calculate_reward()
            next_state = env._get_state()
            done = tuple(env.drone_pos) == env.target
            
            # Q-learning update
            agent.update_q_table(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
            # Render every 50 episodes for progress viewing
            if episode % 50 == 0:
                env.render()
                time.sleep(0.1)
        
        if episode % 50 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")
    
    # Save Q-table
    np.save("q_table.npy", agent.q_table)
    return agent

if __name__ == "__main__":
    trained_agent = train_q_learning(episodes=1000)