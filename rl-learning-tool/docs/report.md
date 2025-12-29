# Interactive RL Learning Tool - Implementation Report

## Abstract

This report presents an interactive web-based tool for learning Reinforcement Learning (RL) algorithms. The tool provides real-time visualization of agent behavior, value functions, and policy updates across multiple environments and algorithms. Users can adjust parameters and observe their effects immediately, making it an effective educational platform for understanding RL concepts.

---

## 1. Introduction to Reinforcement Learning

Reinforcement Learning is a paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards based on its actions and learns to maximize cumulative reward over time.

### Key Components
- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State (s)**: Current situation of the agent
- **Action (a)**: Choices available to the agent
- **Reward (R)**: Feedback signal from the environment
- **Policy (π)**: Strategy for choosing actions

---

## 2. Implemented Environments

### 2.1 GridWorld
A simple navigation task where the agent moves on a grid from start to goal.
- **State Space**: Grid positions (size × size)
- **Actions**: UP, DOWN, LEFT, RIGHT
- **Rewards**: +10 at goal, -0.1 per step
- **Visualization**: Grid with agent position, values, and policy arrows

### 2.2 FrozenLake
Navigate frozen ice to reach a goal while avoiding holes.
- **State Space**: Grid positions (3x3 to 5x5)
- **Actions**: UP, DOWN, LEFT, RIGHT
- **Mode**: Deterministic transitions
- **Rewards**: +1 at goal, 0 otherwise

---

## 3. Implemented Algorithms

### 3.1 Model-Based (Dynamic Programming) Methods

#### Policy Evaluation
Computes V(s) for a given policy using Bellman Expectation Equation.

**Update Rule:**
```
V(s) ← Σ_a π(a|s) × Σ_s' P(s'|s,a) × [R + γV(s')]
```

#### Policy Iteration
Alternates between evaluation and improvement until convergence.

**Steps:**
1. Policy Evaluation: Compute V^π
2. Policy Improvement: π(s) = argmax_a Q(s,a)
3. Repeat until policy is stable

#### Value Iteration
Directly computes optimal values using Bellman Optimality Equation.

**Update Rule:**
```
V(s) ← max_a Σ_s' P(s'|s,a) × [R + γV(s')]
```

### 3.2 Model-Free Methods

#### Monte Carlo
Learns optimal Q(s,a) from complete episode returns with ε-greedy exploration.

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α × (G - Q(s,a))
```
Where G is the discounted return from the episode.

#### TD(0)
Updates after every step using bootstrapping.

**Update Rule:**
```
V(s) ← V(s) + α × [R + γV(s') - V(s)]
```

#### n-step TD
Generalizes TD(0) and Monte Carlo by looking n steps ahead.

**n-step Return:**
```
G = R₁ + γR₂ + ... + γⁿ⁻¹Rₙ + γⁿV(Sₙ)
```

---

## 4. Parameter Adjustment Capabilities

The tool allows real-time adjustment of:

| Parameter | Symbol | Range | Effect |
|-----------|--------|-------|--------|
| Discount Factor | γ | 0.0 - 1.0 | Future reward importance |
| Learning Rate | α | 0.01 - 1.0 | Update speed |
| Exploration Rate | ε | 0.0 - 1.0 | Random action probability |
| n-steps | n | 1 - 10 | TD lookahead |
| Episodes | - | 10 - 200 | Training length |

---

## 5. Visualization Techniques

### 5.1 Environment Visualization
- **Grid environments**: Styled cells with emoji icons
- **Agent position**: robot emoji
- **Labels**: START, GOAL, HOLE labels on special cells
- **State updates**: Real-time agent movement during inference

### 5.2 Training Visualization
- **Value Heatmap**: Color-coded cells (red=low, green=high)
- **Policy Arrows**: Directional arrows (↑↓←→) showing best action
- **Convergence Plot**: Delta values over iterations

### 5.3 Inference Mode
- **Animated Agent**: Watch agent follow learned policy step-by-step
- **Decision Display**: Shows action values at each step
- **Step Counter**: Tracks steps and cumulative reward

---

## 6. Technical Implementation

### 6.1 Technology Stack
- **Frontend/Backend**: Streamlit (Python)
- **Visualization**: HTML/CSS, Pandas DataFrames
- **Algorithms**: NumPy for computations

### 6.2 Project Structure
```
rl-learning-tool/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── environments/          # Environment implementations
│   ├── gridworld.py
│   └── frozenlake.py
└── algorithms/            # RL algorithms
    ├── policy_evaluation.py
    ├── policy_iteration.py
    ├── value_iteration.py
    ├── monte_carlo.py
    ├── td0.py
    └── n_step_td.py
```
