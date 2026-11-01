# Flappy Bird DQN

A PyTorch implementation of a Deep Q-Network (DQN) agent that learns to play Flappy Bird. 

This project uses the `gymnasium` API and `flappy-bird-gymnasium` to simulate the environment, training an agent from scratch to navigate the pipes by balancing exploration (random actions) and exploitation (using the trained neural network).

## Results
After training for **50,000 episodes**, the DQN agent has learned the basic mechanics of the game and can successfully navigate through 8 to 9 pipes before crashing. While it does not yet play perfectly consistently, it demonstrates a clear foundational learning of timing and trajectory. 

Here is a demonstration of the agent in action:

![Trained Flappy Bird Agent](gameplay.gif)

*Note: This GIF is for demonstration purposes. The agent relies purely on its neural network (Exploitation) during this test phase, with the exploration rate (epsilon) decayed to zero.*

## Project Structure

- `agent.py`: The main reinforcement learning loop. Handles environment interaction, action selection, and the training step.
- `dqn.py`: The PyTorch neural network architecture used to approximate Q-values.
- `experience_replay.py`: A cyclic buffer that stores past state transitions so the agent can sample them randomly during training, breaking correlation between consecutive steps.
- `parameters.yaml`: Contains all hyperparameters (learning rate, epsilon decay, batch size, etc.) so you can tweak the training without digging into the code.
- `requirements.txt`: Project dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flappybird-dqn.git
cd flappybird-dqn
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The agent's settings and environment target are defined in `parameters.yaml` under the key `flappybirdv0`.

**To train the model:**
```bash
python agent.py flappybirdv0 --train
```
During training, the agent will run without rendering the game to speed up the process. Best models and training logs are automatically saved to a `runs/` directory.

**To test the trained model (watch it play):**
```bash
python agent.py flappybirdv0
```
This loads the best weights from the `runs/` folder and renders the environment so you can watch the agent play. 

## Hyperparameters

You can adjust the training behavior and model dynamics by editing `parameters.yaml`. Key parameters include:

* `env_id`: Target Gymnasium environment (`FlappyBird-v0`).
* `epsilon_init` / `epsilon_min`: The initial (1.0) and minimum (0.006) exploration rates for the $\epsilon$-greedy policy.
* `decay_step`: Total environment steps over which epsilon decays linearly to its minimum floor.
* `warmup_steps`: Number of initial steps to populate the replay buffer before network optimization begins.
* `replay_memory_size`: Capacity of the circular experience replay buffer (FIFO).
* `mini_batch_size`: Number of transitions sampled per gradient descent update.
* `network_sync_rate`: Step interval for synchronizing policy network weights to the target network.
* `alpha`: Learning rate for the Adam optimizer (`0.0001`).
* `gamma`: Discount factor for future rewards (`0.99`).
* `reward_threshold`: Maximum raw reward limit before completing an episode.
