# 🤖 Interactive RL Learning Tool

An interactive web-based platform for learning and experimenting with Reinforcement Learning algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/SaifAbushanab/Reinforcement-Learning-Tool.git
cd Reinforcement-Learning-Tool/rl-learning-tool

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 🎮 Environments

| Environment | Description |
|-------------|-------------|
| **GridWorld** | Navigate from start to goal |
| **FrozenLake** | Reach goal avoiding holes (deterministic) |

## 📊 Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **Policy Iteration** | Model-Based | Evaluate + improve loop (includes policy evaluation) |
| **Value Iteration** | Model-Based | Direct optimal V computation |
| **Monte Carlo** | Model-Free | Learn from complete episode returns |
| **TD(0)** | Model-Free | One-step bootstrapping |
| **n-step TD** | Model-Free | Generalized TD/MC bridge |

## ⚙️ Parameters

- **γ (gamma)**: Discount factor - how much to value future rewards
- **α (alpha)**: Learning rate - how fast to update estimates
- **ε (epsilon)**: Exploration rate - probability of random action
- **n**: Steps to look ahead (for n-step TD)

## 📁 Project Structure

```
rl-learning-tool/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── environments/          # Environment implementations
│   ├── gridworld.py
│   └── frozenlake.py
└── algorithms/            # RL algorithms
    ├── policy_iteration.py
    ├── value_iteration.py
    ├── monte_carlo.py
    ├── td0.py
    └── n_step_td.py
```

## 📖 Features

- **Interactive Visualization**: See value functions and policies update in real-time
- **Inference Mode**: Watch the trained agent navigate the environment
- **Convergence Plots**: Track algorithm convergence over iterations
- **Parameter Tuning**: Adjust γ, α, ε, and episodes via sliders

## License

MIT
