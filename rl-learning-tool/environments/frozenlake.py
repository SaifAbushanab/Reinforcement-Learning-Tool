"""
FrozenLake Environment

A grid where the agent walks on frozen ice to reach a goal.
Some tiles are holes - falling in ends the episode with no reward.
The ice can be slippery (stochastic) or non-slippery (deterministic).

Map legend:
    S = Start
    F = Frozen (safe)
    H = Hole (game over)
    G = Goal (win!)
"""

import numpy as np


class FrozenLake:
    """
    FrozenLake environment with optional slippery ice.
    
    When slippery=True, the agent only moves in the intended direction
    33% of the time. The other 66% it moves perpendicular.
    """
    
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    # Default 4x4 map
    DEFAULT_MAP = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
    
    def __init__(self, size=4, map_name=None, is_slippery=True, custom_map=None):
        """
        Initialize FrozenLake.
        
        Args:
            size: Grid size (3-8), used if map_name is None
            map_name: "4x4" or "8x8" for preset maps (overrides size)
            is_slippery: If True, ice is slippery (stochastic)
            custom_map: Optional custom map as list of strings
        """
        # Set up the map
        if custom_map:
            self.map = custom_map
        elif map_name == "8x8":
            self.map = self._generate_8x8_map()
        elif map_name == "4x4":
            self.map = self.DEFAULT_MAP
        else:
            # Generate dynamic map based on size
            self.map = self._generate_map(size)
        
        self.size = len(self.map)
        self.n_states = self.size * self.size
        self.n_actions = 4
        self.is_slippery = is_slippery
        
        # Find special positions
        self.start = self._find_char('S')
        self.goal = self._find_char('G')
        self.holes = self._find_all_chars('H')
        
        self.agent_pos = None
        self.done = False
        
        self.name = "FrozenLake"
        self.description = "Reach the goal without falling in holes!"
        
        # Reward settings (can be modified from UI)
        self.goal_reward = 1.0
        self.step_penalty = 0.0  # Default no step penalty for FrozenLake
        
    def _generate_8x8_map(self):
        """Generate a larger 8x8 map."""
        return [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ]
    
    def _generate_map(self, size):
        """
        Generate a map of given size with appropriate holes.
        Start at top-left, Goal at bottom-right.
        """
        import random
        
        # Create base frozen map
        grid = [['F' for _ in range(size)] for _ in range(size)]
        
        # Set start and goal
        grid[0][0] = 'S'
        grid[size-1][size-1] = 'G'
        
        # Calculate number of holes based on size
        # Roughly 20-25% of cells (excluding start and goal)
        n_holes = max(1, (size * size - 2) // 4)
        
        # Place holes randomly (avoiding start, goal, and ensuring path exists)
        random.seed(42)  # Fixed seed for reproducibility
        available = [(r, c) for r in range(size) for c in range(size) 
                     if (r, c) not in [(0, 0), (size-1, size-1)]]
        
        # Avoid blocking the path - keep diagonal somewhat clear
        safe_diagonal = [(i, i) for i in range(1, size-1)]
        available = [pos for pos in available if pos not in safe_diagonal]
        
        holes = random.sample(available, min(n_holes, len(available)))
        for r, c in holes:
            grid[r][c] = 'H'
        
        return [''.join(row) for row in grid]
    
    def _find_char(self, char):
        """Find position of a character in the map."""
        for row in range(self.size):
            for col in range(self.size):
                if self.map[row][col] == char:
                    return (row, col)
        return (0, 0)
    
    def _find_all_chars(self, char):
        """Find all positions of a character in the map."""
        positions = []
        for row in range(self.size):
            for col in range(self.size):
                if self.map[row][col] == char:
                    positions.append((row, col))
        return positions
    
    def reset(self):
        """Reset environment to start."""
        self.agent_pos = list(self.start)
        self.done = False
        return self._get_state()
    
    def step(self, action):
        """Take an action. Returns (next_state, reward, done, info)."""
        if self.done:
            return self._get_state(), 0, True, {}
        
        # If slippery, the actual action might be different
        if self.is_slippery:
            actual_action = self._get_slippery_action(action)
        else:
            actual_action = action
        
        # Move the agent
        row, col = self.agent_pos
        new_row, new_col = self._get_next_pos(row, col, actual_action)
        
        # Check bounds
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self.agent_pos = [new_row, new_col]
        
        # Check outcome
        current_tile = self.map[self.agent_pos[0]][self.agent_pos[1]]
        
        if current_tile == 'H':
            # Fell in a hole!
            self.done = True
            reward = 0.0
        elif current_tile == 'G':
            # Reached the goal!
            self.done = True
            reward = self.goal_reward
        else:
            # Still on ice
            reward = -self.step_penalty
        
        return self._get_state(), reward, self.done, {"slipped": actual_action != action}
    
    def _get_slippery_action(self, intended_action):
        """When slippery, randomly choose actual movement direction."""
        # 33% intended, 33% left of intended, 33% right of intended
        if intended_action in [self.UP, self.DOWN]:
            options = [intended_action, self.LEFT, self.RIGHT]
        else:
            options = [intended_action, self.UP, self.DOWN]
        
        return np.random.choice(options)
    
    def _get_next_pos(self, row, col, action):
        """Get next position given current position and action."""
        if action == self.UP:
            return row - 1, col
        elif action == self.DOWN:
            return row + 1, col
        elif action == self.LEFT:
            return row, col - 1
        elif action == self.RIGHT:
            return row, col + 1
        return row, col
    
    def _get_state(self):
        """Convert position to state number."""
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def state_to_pos(self, state):
        """Convert state number to (row, col)."""
        return (state // self.size, state % self.size)
    
    def get_valid_actions(self, state=None):
        """All 4 actions are always available."""
        return list(range(self.n_actions))
    
    def get_transition_prob(self, state, action):
        """
        Get transition probabilities for planning algorithms.
        
        Returns list of (probability, next_state, reward, done) tuples.
        """
        row, col = self.state_to_pos(state)
        
        if self.is_slippery:
            # Three possible outcomes
            if action in [self.UP, self.DOWN]:
                actions = [action, self.LEFT, self.RIGHT]
            else:
                actions = [action, self.UP, self.DOWN]
            
            transitions = []
            for a in actions:
                new_row, new_col = self._get_next_pos(row, col, a)
                
                # Check bounds
                if not (0 <= new_row < self.size and 0 <= new_col < self.size):
                    new_row, new_col = row, col
                
                next_state = new_row * self.size + new_col
                tile = self.map[new_row][new_col]
                
                if tile == 'H':
                    reward, done = 0.0, True
                elif tile == 'G':
                    reward, done = self.goal_reward, True
                else:
                    reward, done = -self.step_penalty, False
                
                transitions.append((1/3, next_state, reward, done))
            
            return transitions
        else:
            # Deterministic
            new_row, new_col = self._get_next_pos(row, col, action)
            
            if not (0 <= new_row < self.size and 0 <= new_col < self.size):
                new_row, new_col = row, col
            
            next_state = new_row * self.size + new_col
            tile = self.map[new_row][new_col]
            
            if tile == 'H':
                reward, done = 0.0, True
            elif tile == 'G':
                reward, done = self.goal_reward, True
            else:
                reward, done = -self.step_penalty, False
            
            return [(1.0, next_state, reward, done)]
    
    def render_data(self):
        """Get visualization data."""
        return {
            "size": self.size,
            "map": self.map,
            "agent_pos": tuple(self.agent_pos) if self.agent_pos else self.start,
            "goal": self.goal,
            "start": self.start,
            "holes": self.holes,
            "is_slippery": self.is_slippery,
            "done": self.done,
        }
