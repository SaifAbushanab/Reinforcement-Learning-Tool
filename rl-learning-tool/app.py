"""
Interactive RL Learning Tool - Main Streamlit Application

A web-based platform for learning and experimenting with 
Reinforcement Learning algorithms through interactive visualization.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import time
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="RL Learning Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import environments and algorithms
from environments import ENVIRONMENTS
from algorithms import ALGORITHMS, ALGORITHM_INFO

# ================================================================
# CONCEPT EXPLANATIONS
# ================================================================

ENV_DESCRIPTIONS = {
    "GridWorld": "Navigate from start to goal. The agent learns the shortest path.",
    "FrozenLake": "Walk on frozen ice to reach the goal. Watch out for holes!",
}

ALGO_DETAILS = {
    "Policy Iteration": {
        "update_rule": "1) V(s) ← Σ π(a|s)[R + γV(s')]  2) π(s) = argmax Q",
        "key_idea": "Alternate between evaluating V(s) and improving policy until optimal.",
    },
    "Value Iteration": {
        "update_rule": "V(s) ← max_a [R + γV(s')]",
        "key_idea": "Directly compute optimal values with max over actions.",
    },
    "Monte Carlo": {
        "update_rule": "Q(s,a) ← Q(s,a) + α(G - Q(s,a))",
        "key_idea": "Learn Q-values from complete episode returns with exploration.",
    },
    "TD(0) Prediction": {
        "update_rule": "V(s) ← V(s) + α[R + γV(s') - V(s)]",
        "key_idea": "Bootstrap: update using next state estimate.",
    },
    "n-step TD": {
        "update_rule": "G = R₁ + γR₂ + ... + γⁿV(Sₙ)",
        "key_idea": "Look n steps ahead before bootstrapping.",
    },
}

# ================================================================
# VISUALIZATION FUNCTIONS
# ================================================================

def get_value_color(value, min_val, max_val):
    """Get background color based on value (red=low, green=high)."""
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    
    # Red to Green gradient
    red = int(255 * (1 - normalized))
    green = int(255 * normalized)
    return f"rgb({red}, {green}, 100)"


def render_grid_environment(env_data, values=None, policy=None, env_type="GridWorld", 
                            agent_pos_override=None, highlight_state=None):
    """Render grid environment with value heatmap and policy arrows."""
    size = env_data["size"]
    agent_pos = agent_pos_override if agent_pos_override else env_data["agent_pos"]
    
    # Action arrows
    arrows = ["↑", "↓", "←", "→"]
    
    # Get value range for color coding
    if values is not None:
        min_val, max_val = np.min(values), np.max(values)
    else:
        min_val, max_val = 0, 1
    
    for row in range(size):
        cols = st.columns(size)
        for col in range(size):
            pos = (row, col)
            state = row * size + col
            
            # Determine cell content with labels
            label = ""
            if pos == agent_pos:
                icon = "🤖"
            elif env_type == "GridWorld":
                if pos == env_data.get("goal"):
                    icon = "🎯"
                    label = "GOAL"
                elif pos == env_data.get("start"):
                    icon = "🏁"
                    label = "START"
                elif pos in [tuple(w) for w in env_data.get("walls", [])]:
                    icon = "⬛"
                else:
                    icon = "⬜"
            elif env_type == "FrozenLake":
                tile = env_data["map"][row][col]
                if tile == 'H':
                    icon = "🕳️"
                    label = "HOLE"
                elif tile == 'G':
                    icon = "🎯"
                    label = "GOAL"
                elif tile == 'S':
                    icon = "🏁"
                    label = "START"
                else:
                    icon = "🧊"
            else:
                icon = "⬜"
            
            # Build cell text with label
            if label:
                cell_text = f"{icon}\n{label}"
            else:
                cell_text = icon
            
            # Add policy arrow
            if policy is not None and len(policy) > state:
                if max(policy[state]) > 0:
                    best_action = np.argmax(policy[state])
                    cell_text = arrows[best_action] + " " + cell_text
            
            # Render the cell with styling
            with cols[col]:
                if values is not None and len(values) > state:
                    value = values[state]
                    color = get_value_color(value, min_val, max_val)
                    cell_text += f"\n{value:.1f}"
                    
                    # Highlight current state in inference mode
                    if highlight_state == state:
                        st.markdown(f"<div style='background-color: yellow; padding: 5px; border-radius: 5px; text-align: center;'><b>{cell_text}</b></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color: {color}; padding: 5px; border-radius: 5px; text-align: center;'><b>{cell_text}</b></div>", unsafe_allow_html=True)
                else:
                    # Initial render without values - still use styled div
                    st.markdown(f"<div style='background-color: #374151; padding: 5px; border-radius: 5px; text-align: center;'><b>{cell_text}</b></div>", unsafe_allow_html=True)


def plot_convergence(deltas):
    """Plot convergence curve (delta over iterations)."""
    if len(deltas) > 0:
        df = pd.DataFrame({
            "Iteration": range(1, len(deltas) + 1), 
            "Delta (Value Change)": deltas
        })
        st.line_chart(df.set_index("Iteration"))


def plot_rewards(rewards):
    """Plot reward curve."""
    if len(rewards) > 0:
        df = pd.DataFrame({"Episode": range(1, len(rewards) + 1), "Reward": rewards})
        st.line_chart(df.set_index("Episode"))


def render_q_table(Q, max_display=50):
    """Render Q-table."""
    if Q is None or len(Q) == 0:
        return
    
    n_states = min(len(Q), max_display)
    n_actions = Q.shape[1]
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"][:n_actions]
    
    df = pd.DataFrame(Q[:n_states], columns=action_names)
    df.index.name = "State"
    st.dataframe(df.style.background_gradient(cmap="RdYlGn", axis=None))


def render_action_values(Q_or_policy, state, action_names=["↑ UP", "↓ DOWN", "← LEFT", "→ RIGHT"]):
    """Display action values or probabilities for current state."""
    if Q_or_policy is None or len(Q_or_policy) <= state:
        return
    
    values = Q_or_policy[state]
    best_action = np.argmax(values)
    
    st.write("**Decision at Current State:**")
    for i, (name, val) in enumerate(zip(action_names, values)):
        if i == best_action:
            st.markdown(f"✅ **{name}: {val:.3f}** (chosen)")
        else:
            st.write(f"   {name}: {val:.3f}")


# ================================================================
# MAIN APPLICATION
# ================================================================

def main():
    st.title("🤖 Interactive RL Learning Tool")
    st.caption("Learn Reinforcement Learning algorithms visually!")
    
    # ================================
    # SIDEBAR
    # ================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment
        st.subheader("1️⃣ Environment")
        env_name = st.selectbox("Select Environment", list(ENVIRONMENTS.keys()))
        st.caption(ENV_DESCRIPTIONS.get(env_name, ""))
        
        # Grid size slider (same for both environments)
        grid_size = st.slider("Grid Size", 3, 5, 4)
        
        # FrozenLake is always deterministic (no slippery ice)
        is_slippery = False
        
        st.divider()
        
        # Algorithm
        st.subheader("2️⃣ Algorithm")
        algo_name = st.selectbox("Select Algorithm", list(ALGORITHMS.keys()))
        algo_info = ALGORITHM_INFO.get(algo_name, {})
        st.info(algo_info.get("description", ""))
        
        st.divider()
        
        # Parameters
        st.subheader("3️⃣ Parameters")
        
        gamma = st.slider("γ (Discount)", 0.0, 1.0, 0.99, 0.01,
                         help="How much to value future rewards")
        
        if algo_info.get("uses_alpha", True):
            alpha = st.slider("α (Learning Rate)", 0.01, 1.0, 0.1, 0.01,
                             help="How fast to update values")
        else:
            alpha = 0.1
        
        if algo_info.get("uses_epsilon", True):
            epsilon = st.slider("ε (Exploration)", 0.0, 1.0, 0.1, 0.01,
                               help="Probability of random action")
        else:
            epsilon = 0.1
        
        if "n-step" in algo_name:
            n_steps = st.slider("n (Steps)", 1, 10, 3)
        else:
            n_steps = 3
        
        # Episodes only for model-free algorithms
        if not algo_info.get("model_based", False):
            n_episodes = st.slider("Episodes", 10, 200, 100, 10)
        else:
            n_episodes = 100  # Default value, not used for model-based
        
        # Reward shaping
        st.markdown("**Reward Settings:**")
        goal_reward = st.slider("Goal Reward", 1.0, 20.0, 10.0, 1.0,
                               help="Reward for reaching the goal")
        step_penalty = st.slider("Step Penalty", -1.0, 0.0, -0.1, 0.01,
                                help="Negative reward per step (encourages shorter paths)")
        
        st.divider()
        
        # Controls
        st.subheader("4️⃣ Controls")
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶️ Train", use_container_width=True)
        with col2:
            reset_btn = st.button("🔄 Reset", use_container_width=True)
        
        # Inference button (only show after training)
        if st.session_state.get("training_complete", False):
            inference_btn = st.button("🎯 Run Inference", use_container_width=True)
        else:
            inference_btn = False
        
        # Fixed slow animation speed
        delay = 0.5
    
    # ================================
    # MAIN AREA
    # ================================
    
    # Session state initialization
    if "running" not in st.session_state:
        st.session_state.running = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "training_complete" not in st.session_state:
        st.session_state.training_complete = False
    if "deltas" not in st.session_state:
        st.session_state.deltas = []
    
    if reset_btn:
        st.session_state.running = False
        st.session_state.results = None
        st.session_state.training_complete = False
        st.session_state.deltas = []
        st.rerun()
    
    # Create environment with reward settings
    if env_name == "GridWorld":
        env = ENVIRONMENTS[env_name](size=grid_size)
        env.goal_reward = goal_reward
        env.step_penalty = abs(step_penalty)  # step_penalty is negative in UI
    else:  # FrozenLake
        env = ENVIRONMENTS[env_name](
            size=grid_size,
            is_slippery=is_slippery
        )
        env.goal_reward = goal_reward
        env.step_penalty = abs(step_penalty)  # step_penalty is negative in UI
    
    # Layout
    main_col, side_col = st.columns([2, 1])
    
    with main_col:
        st.subheader(f"🎮 {env_name}")
        env_placeholder = st.empty()
        
        # Initial render
        env.reset()
        with env_placeholder.container():
            if env_name == "GridWorld":
                render_grid_environment(env.render_data(), env_type="GridWorld")
            else:  # FrozenLake
                render_grid_environment(env.render_data(), env_type="FrozenLake")
    
    with side_col:
        st.subheader("📊 Training Progress")
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        convergence_placeholder = st.empty()
    
    # ================================
    # TRAINING
    # ================================
    if start_btn:
        st.session_state.running = True
        st.session_state.training_complete = False
        st.session_state.deltas = []
        
        algorithm = ALGORITHMS[algo_name]
        algo_args = {"gamma": gamma}
        
        if algo_info.get("uses_alpha", True):
            algo_args["alpha"] = alpha
        if algo_info.get("uses_epsilon", True):
            algo_args["epsilon"] = epsilon
        if "n_episodes" in algorithm.__code__.co_varnames:
            algo_args["n_episodes"] = n_episodes
        if "n_steps" in algorithm.__code__.co_varnames:
            algo_args["n_steps"] = n_steps
        
        progress_bar = st.progress(0)
        deltas = []
        
        try:
            for result in algorithm(env, **algo_args):
                if not st.session_state.running:
                    break
                
                episode = result.get("episode", result.get("iteration", 0))
                max_iter = result.get("iteration", n_episodes) if algo_info.get("model_based", False) else n_episodes
                progress_bar.progress(min(1.0, episode / max(max_iter, 1)))
                
                # Track deltas for convergence plot
                if "delta" in result:
                    deltas.append(result["delta"])
                
                # Update visualization with heatmap
                with env_placeholder.container():
                    values = result.get("values")
                    policy = result.get("policy")
                    
                    if env_name == "GridWorld":
                        render_grid_environment(env.render_data(), values, policy, "GridWorld")
                    else:  # FrozenLake
                        render_grid_environment(env.render_data(), values, policy, "FrozenLake")
                
                # Update metrics
                with metrics_placeholder.container():
                    # Show "Iteration" for model-based, "Episode" for model-free
                    if "iteration" in result:
                        st.metric("Iteration", result["iteration"])
                    else:
                        st.metric("Episode", result.get("episode", 0))
                    if "episode_reward" in result:
                        st.metric("Reward", f"{result['episode_reward']:.2f}")
                    if "delta" in result:
                        st.metric("Delta", f"{result['delta']:.6f}")
                    if result.get("converged", False):
                        st.success("✅ Converged!")
                
                # Update reward chart (model-free)
                if "episode_rewards" in result:
                    with chart_placeholder.container():
                        st.caption("Episode Rewards")
                        plot_rewards(result["episode_rewards"])
                
                # Update convergence chart (model-based)
                if len(deltas) > 1:
                    with convergence_placeholder.container():
                        st.caption("Convergence (Delta)")
                        plot_convergence(deltas)
                
                time.sleep(delay)
                st.session_state.results = result
            
            st.session_state.training_complete = True
            st.session_state.deltas = deltas
            st.success("✅ Training complete! Click '🎯 Run Inference' to see the agent in action.")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # ================================
    # INFERENCE MODE
    # ================================
    if inference_btn and st.session_state.results:
        st.subheader("🎯 Inference: Agent Following Learned Policy")
        
        result = st.session_state.results
        policy = result.get("policy")
        Q = result.get("Q")
        values = result.get("values")
        
        # Reset environment for inference
        state = env.reset()  # Get initial state from reset
        
        inference_placeholder = st.empty()
        decision_placeholder = st.empty()
        step_info = st.empty()
        
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 50  # Prevent infinite loops
        
        while not done and step_count < max_steps:
            # Get action from policy or Q-table
            if Q is not None and len(Q) > state:
                action = np.argmax(Q[state])
                action_values = Q[state]
            elif policy is not None and len(policy) > state:
                action = np.argmax(policy[state])
                action_values = policy[state]
            else:
                action = 0
                action_values = None
            
            # Render current state
            with inference_placeholder.container():
                agent_row = state // env.size
                agent_col = state % env.size
                if env_name == "GridWorld":
                    render_grid_environment(
                        env.render_data(), values, policy, "GridWorld",
                        agent_pos_override=(agent_row, agent_col),
                        highlight_state=state
                    )
                else:
                    render_grid_environment(
                        env.render_data(), values, policy, "FrozenLake",
                        agent_pos_override=(agent_row, agent_col),
                        highlight_state=state
                    )
            
            # Show decision-making
            with decision_placeholder.container():
                render_action_values(Q if Q is not None else policy, state)
            
            # Show step info
            with step_info.container():
                st.write(f"**Step {step_count + 1}** | State: {state} | Total Reward: {total_reward:.2f}")
            
            time.sleep(delay)
            
            # Take action and update state
            next_state, reward, done, _ = env.step(action)
            state = next_state  # Update state for next iteration
            total_reward += reward
            step_count += 1
        
        if done:
            st.success(f"🎉 Goal reached in {step_count} steps! Total Reward: {total_reward:.2f}")
        else:
            st.warning(f"⚠️ Max steps ({max_steps}) reached. Total Reward: {total_reward:.2f}")
    
    # ================================
    # Q-TABLE AND ALGORITHM INFO
    # ================================
    if st.session_state.results and "Q" in st.session_state.results:
        st.subheader("📋 Q-Table")
        render_q_table(st.session_state.results["Q"])
    
    # Algorithm info
    st.divider()
    algo_detail = ALGO_DETAILS.get(algo_name, {})
    with st.expander("📖 Algorithm Details"):
        st.write(f"**{algo_name}**")
        st.code(algo_detail.get("update_rule", ""))
        st.write(algo_detail.get("key_idea", ""))


if __name__ == "__main__":
    main()
