from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.ibuprofen_env import IbuprofenEnv
from sb3.callback_sb3 import RewardLoggingCallback
from sb3.ibuprofen_env_sb3.scripts.optimize_sb3_ibuprofen import get_best_params

def train_ppo_model(env, best_params, total_timesteps, callback=None):
    """
    Train a PPO model using the specified environment and hyperparameters.

    Args:
        env (DummyVecEnv): The training environment.
        best_params (dict): Dictionary containing optimized hyperparameters for PPO.
        total_timesteps (int): Total timesteps for training the model.
        callback (BaseCallback, optional): A callback for logging or monitoring training.

    Returns:
        PPO: The trained PPO model.
    """
    # Initialize the PPO model with the optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=best_params["learning_rate"],
        gamma=best_params["gamma"],
        n_epochs=best_params["n_epochs"],
        ent_coef=best_params["ent_coef"],
        batch_size=best_params["batch_size"],
        n_steps=best_params["n_steps"],
        gae_lambda=best_params["gae_lambda"],
        clip_range=best_params["clip_range"],
        verbose=1,
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model


def dynamic_training_loop(model, env, initial_horizon, max_horizon, horizon_increment, num_episodes):
    """
    Run a training loop with a dynamically adjusted time horizon.

    Args:
        model (PPO): The trained PPO model.
        env (DummyVecEnv): The environment used for training and evaluation.
        initial_horizon (int): Initial time horizon for the episodes.
        max_horizon (int): Maximum time horizon for the episodes.
        horizon_increment (int): Increment for time horizon after each episode.
        num_episodes (int): Number of episodes to train.

    Returns:
        list: A history of total rewards for each episode.
    """
    reward_history = []

    # Initialize the dynamic time horizon
    time_horizon = initial_horizon

    # Training loop over the specified number of episodes
    for episode in range(num_episodes):
        # Update the time horizon dynamically
        time_horizon = min(max_horizon, initial_horizon + episode * horizon_increment)

        # Reset the environment at the beginning of the episode
        state = env.reset()

        total_reward = 0
        plasma_concentration_history = []

        # Run the episode for the current time horizon
        for t in range(time_horizon):
            # Predict the action using the PPO model
            action, _ = model.predict(state, deterministic=False)

            # Take a step in the environment
            new_state, reward, done, infos = env.step(action)
            plasma_concentration_history.append(new_state[0])  # Store plasma concentration

            # Accumulate the reward
            total_reward += reward

            # Update the current state
            state = new_state

            # Check if the episode is done
            if done:
                break

        # Append the total reward for the episode to the history
        reward_history.append(total_reward)

        # Log progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}, Time Horizon = {time_horizon}")

    return reward_history


if __name__ == "__main__":

    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])
    callback = RewardLoggingCallback()

    best_params = get_best_params(1)

    # Train the PPO model
    final_model = train_ppo_model(env, best_params, total_timesteps=24000, callback=callback)

    # Run the dynamic training loop
    reward_history = dynamic_training_loop(
        model=final_model,
        env=env,
        initial_horizon=6,
        max_horizon=24,
        horizon_increment=2,
        num_episodes=1000
    )
