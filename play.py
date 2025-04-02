import time
import numpy as np
from env import DroneGridEnv

def play_with_qtable(qtable_path="q_table.npy"):
    env = DroneGridEnv()
    q_table = np.load(qtable_path)
    
    while True:
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 100:
            drone_x, drone_y, target_x, target_y = state
            action = np.argmax(q_table[drone_x, drone_y, target_x, target_y])
            direction = ['up', 'down', 'left', 'right'][action]
            
            env.move(direction)
            state = env._get_state()
            reward = env.calculate_reward()
            done = tuple(env.drone_pos) == env.target
            
            env.render()
            print(f"Step: {step}, Action: {direction}, Position: {env.drone_pos}, Reward: {reward:.1f}")
            
            total_reward += reward
            step += 1
            time.sleep(0.3)
        
        print(f"Episode complete! Total reward: {total_reward:.1f}")
        print("Restarting in 3 seconds...")
        time.sleep(3)

if __name__ == "__main__":
    play_with_qtable()