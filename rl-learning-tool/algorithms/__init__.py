# Algorithms package - all algorithms as single files

from .policy_iteration import policy_iteration
from .value_iteration import value_iteration
from .monte_carlo import mc_prediction, mc_control
from .td0 import td_prediction
from .n_step_td import n_step_td

# Map of algorithm names to functions
ALGORITHMS = {
    "Policy Iteration": policy_iteration,
    "Value Iteration": value_iteration,
    "Monte Carlo": mc_control,
    "TD(0) Prediction": td_prediction,
    "n-step TD": n_step_td,
}

# Algorithm info for UI
ALGORITHM_INFO = {
    "Policy Iteration": {
        "description": "Alternates between policy evaluation and improvement to find optimal policy.",
        "type": "control",
        "model_based": True,
        "uses_alpha": False,
        "uses_epsilon": False,
    },
    "Value Iteration": {
        "description": "Computes optimal value function by taking max over actions.",
        "type": "control",
        "model_based": True,
        "uses_alpha": False,
        "uses_epsilon": False,
    },
    "Monte Carlo": {
        "description": "Learns optimal policy by running complete episodes and averaging returns.",
        "type": "control",
        "uses_alpha": True,
        "uses_epsilon": True,
    },
    "TD(0) Prediction": {
        "description": "Updates value estimates using one-step bootstrapping.",
        "type": "prediction",
        "uses_alpha": True,
        "uses_epsilon": True,
    },
    "n-step TD": {
        "description": "Generalizes TD(0) and Monte Carlo by looking n steps ahead.",
        "type": "prediction",
        "uses_alpha": True,
        "uses_epsilon": True,
    },
}
