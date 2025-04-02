import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class DroneGridEnv:
    def __init__(self):
        # define the grid as done in the report of 8x8
        self.grid_size = 8
        self.target = (6, 7)
        self.start_pos = (1, 1)
        self.obstacles = {(3, 3), (4, 5), (6, 2), (7, 1), (3, 7), (1, 0)}
        self.drone_pos = list(self.start_pos)
        self.path_history = []
        
        self.fig, self.ax = None, None
        self.colors = {
            'drone': '#3498db',
            'target': '#2ecc71', 
            'obstacle': '#e74c3c',
            'grid': '#ecf0f1',
            'path': '#9b59b6'
        }

    def reset(self):
        self.drone_pos = list(self.start_pos)
        self.path_history = [tuple(self.drone_pos)]
        return self._get_state()

    def _get_state(self):
        return (*self.drone_pos, *self.target)

    def move(self, direction):
        new_pos = self.drone_pos.copy()
        
        if direction == 'up': new_pos[0] -= 1
        elif direction == 'down': new_pos[0] += 1
        elif direction == 'left': new_pos[1] -= 1
        elif direction == 'right': new_pos[1] += 1
        
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size and
            tuple(new_pos) not in self.obstacles):
            self.drone_pos = new_pos
            self.path_history.append(tuple(self.drone_pos))
            return True
        return False

    def calculate_reward(self):
        distance = np.linalg.norm(np.array(self.target) - np.array(self.drone_pos))
        
        if tuple(self.drone_pos) == self.target:
            return 100
        if tuple(self.drone_pos) in self.obstacles:
            return -50
            
        movement_penalty = -0.2
        distance_penalty = -distance * 0.5
        visit_count = self.path_history.count(tuple(self.drone_pos))
        exploration_bonus = 0.5 if visit_count < 2 else -0.3 * visit_count
        
        return distance_penalty + movement_penalty + exploration_bonus

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_aspect('equal')
            self.ax.set_facecolor(self.colors['grid'])
            self.ax.grid(True, color='white')
            
            # Draw obstacles (col, row)
            for obs in self.obstacles:
                self.ax.add_patch(Rectangle(
                    (obs[1], obs[0]), 1, 1, 
                    color=self.colors['obstacle'], alpha=0.7
                ))
                self.ax.text(
                    obs[1]+0.5, obs[0]+0.5, 'X',
                    ha='center', va='center',
                    color='white', weight='bold'
                )
            
            # Draw target (col, row)
            self.ax.add_patch(Rectangle(
                (self.target[1], self.target[0]), 1, 1,
                color=self.colors['target'], alpha=0.7
            ))
            self.ax.text(
                self.target[1]+0.5, self.target[0]+0.5, 'T',
                ha='center', va='center',
                color='white', weight='bold'
            )
            
            # Initialize drone (col, row)
            self.drone_artist = Circle(
                (self.drone_pos[1]+0.5, self.drone_pos[0]+0.5),
                0.4, color=self.colors['drone']
            )
            self.ax.add_patch(self.drone_artist)
            
            self.path_line, = self.ax.plot(
                [y+0.5 for x,y in self.path_history],
                [x+0.5 for x,y in self.path_history],
                color=self.colors['path'], linestyle=':', marker='o', markersize=4
            )
            
            plt.title("Drone Navigation (Corrected Coordinates)")
        
        # Upating visualization
        self.drone_artist.center = (self.drone_pos[1]+0.5, self.drone_pos[0]+0.5)
        self.path_line.set_data(
            [y+0.5 for x,y in self.path_history],
            [x+0.5 for x,y in self.path_history]
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            plt.close(self.fig)