# Environments package
from .gridworld import GridWorld
from .frozenlake import FrozenLake

# Map of environment names to classes
ENVIRONMENTS = {
    "GridWorld": GridWorld,
    "FrozenLake": FrozenLake,
}
