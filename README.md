# Implementing Proximal Policy Optimization (PPO) Algorithm

This project simulates the delivery of ibuprofen through a reinforcement learning (RL) agent. The agent interacts with a custom environment that models the pharmacokinetics of ibuprofen in the human body. The goal of the agent is to maintain plasma concentration within a therapeutic range while avoiding toxicity. The project uses **Proximal Policy Optimization (PPO)** for training the agent.

A PPO algorithm is constructed in the directory ```custom_PPO```, and the directory ```stable_baselines3``` uses stable_baselines3 for the PPO. Moreover, three environments are used (one is a customized drug delivery environment for ibuprofen, CartPole-v1 & LunarLander-v3).

## Installation

### 1. Clone the repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/xuenbeichin/Reinforcment-Learning--PPO--Implementation-on-Ibuprofen-Delivery-.git
```

### 2. Install Dependencies
This project requires the following libraries:

numpy: Used for array and matrix operations.
torch: PyTorch for building and training the neural networks.
gym: The OpenAI Gym library for building the environment.
matplotlib: For plotting the results and training performance.

Use pip to install all necessary dependencies listed in requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Custom PPO Script
The directory custom_PPO is coded without using any external PPO model. 

**Training the PPO Agent**
To train the PPO agent, run the train_ppo.py script:
```python
trained_agent, rewards = train_agent(n_episodes=1000, n_trials=20) # you can customize your values

plot_training(rewards)
```
This will:
- Initialize the IbuprofenEnv environment and the PPO agent.
- Train the agent for a specified number of episodes.
- Save the agent's trained model.

**Evaluating the PPO Agent**
To evaluate the trained PPO agent, run the evaluate_agent.py script:

``` python
# Perform hyperparameter tuning to find the best parameters
best_params = run_hyperparameter_tuning(n_trials=20)

# Initialize the environment and the PPO agent
env = IbuprofenEnv()
state_dim = env.observation_space.shape[0]  # Dimension of the state space
action_dim = env.action_space.n  # Number of possible actions

# Create a PPO agent with the optimized hyperparameters
agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, **best_params)

# Evaluate the trained agent
trajectory = evaluate_agent(agent, env)

plot_evaluation(trajectory)
```

This will:
- Load the trained PPO agent.
- Evaluate its performance in the environment.
- Plot the results, showing the plasma concentration over time and how well the agent performs.

### Using stable_baselines3

**Notes**:

There are two jupyter notebooks with a full run for both custom PPO and the use of stable_baselines3. Visualizations are provided too.




