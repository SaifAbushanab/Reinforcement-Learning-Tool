"""
GridWorld Environment

A simple grid-based environment where an agent navigates from start to goal.
The agent can move in 4 directions: UP, DOWN, LEFT, RIGHT.

This is the classic RL environment used to teach:
- State representation
- Action effects
- Reward structure
- Episode termination
"""

import numpy as np


class GridWorld:
    """
    A simple GridWorld environment.
    
    The agent starts at a position and must reach the goal.
    Walls block movement, and each step gives a small negative reward
    to encourage finding the shortest path.
    """
    
    # Action definitions (makes code more readable)
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    def __init__(self, size=5, walls=None, start=None, goal=None):
        """
        Initialize the GridWorld.
        
        Args:
            size: Grid size (size x size)
            walls: List of (row, col) positions that are walls
            start: Starting position (row, col)
            goal: Goal position (row, col)
        """
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        
        # Default positions
        self.start = start if start else (0, 0)
        self.goal = goal if goal else (size - 1, size - 1)
        self.walls = set(walls) if walls else set()
        
        # Current agent position
        self.agent_pos = None
        self.done = False
        
        # Reward settings (can be modified from UI)
        self.goal_reward = 10.0
        self.step_penalty = 0.1
        
        # For visualization
        self.name = "GridWorld"
        self.description = "Navigate from start (green) to goal (red)"
        
    def reset(self):
        """Reset the environment and return initial state."""
        self.agent_pos = list(self.start)
        self.done = False
        return self._get_state()
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
            
        Returns:
            next_state: The new state after taking the action
            reward: The reward received
            done: Whether the episode is finished
            info: Additional information (for debugging)
        """
        if self.done:
            return self._get_state(), 0, True, {"message": "Episode already done"}
        
        # Calculate new position based on action
        row, col = self.agent_pos
        
        if action == self.UP:
            new_row, new_col = row - 1, col
        elif action == self.DOWN:
            new_row, new_col = row + 1, col
        elif action == self.LEFT:
            new_row, new_col = row, col - 1
        elif action == self.RIGHT:
            new_row, new_col = row, col + 1
        else:
            new_row, new_col = row, col  # Invalid action, stay in place
        
        # Check if new position is valid (within bounds and not a wall)
        if self._is_valid_position(new_row, new_col):
            self.agent_pos = [new_row, new_col]
        
        # Check if reached goal
        if tuple(self.agent_pos) == self.goal:
            self.done = True
            reward = self.goal_reward  # Configurable goal reward
        else:
            reward = -self.step_penalty  # Configurable step penalty
        
        return self._get_state(), reward, self.done, {}
    
    def _is_valid_position(self, row, col):
        """Check if a position is valid (in bounds and not a wall)."""
        # Check bounds
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        # Check walls
        if (row, col) in self.walls:
            return False
        return True
    
    def _get_state(self):
        """Convert position to state number."""
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def state_to_pos(self, state):
        """Convert state number to (row, col) position."""
        return (state // self.size, state % self.size)
    
    def get_valid_actions(self, state=None):
        """Get list of valid actions from current or given state."""
        return list(range(self.n_actions))  # All actions always valid (may just not move)
    
    def get_transition_prob(self, state, action):
        """
        Get transition probabilities for dynamic programming methods.
        
        Returns list of (probability, next_state, reward, done) tuples.
        GridWorld is deterministic, so probability is always 1.0.
        """
        row, col = self.state_to_pos(state)
        
        # Calculate next position
        if action == self.UP:
            new_row, new_col = row - 1, col
        elif action == self.DOWN:
            new_row, new_col = row + 1, col
        elif action == self.LEFT:
            new_row, new_col = row, col - 1
        elif action == self.RIGHT:
            new_row, new_col = row, col + 1
        else:
            new_row, new_col = row, col
        
        # Check validity
        if not self._is_valid_position(new_row, new_col):
            new_row, new_col = row, col  # Stay in place
        
        next_state = new_row * self.size + new_col
        
        # Determine reward and done (use configurable values)
        if (new_row, new_col) == self.goal:
            reward = self.goal_reward
            done = True
        else:
            reward = -self.step_penalty
            done = False
        
        return [(1.0, next_state, reward, done)]
    
    def render_data(self):
        """
        Get data needed for visualization.
        
        Returns a dictionary with all info needed to draw the environment.
        """
        return {
            "size": self.size,
            "agent_pos": tuple(self.agent_pos) if self.agent_pos else self.start,
            "goal": self.goal,
            "start": self.start,
            "walls": list(self.walls),
            "done": self.done,
        }
