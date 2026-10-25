# Flappy Bird DQN

A PyTorch implementation of a Deep Q-Network (DQN) agent that learns to play Flappy Bird. 

This project uses the `gymnasium` API and `flappy-bird-gymnasium` to simulate the environment, training an agent from scratch to navigate the pipes by balancing exploration (random actions) and exploitation (using the trained neural network).

## Results
After training for **50,000 episodes**, the DQN agent successfully learned the optimal timing and trajectory to navigate through the pipes consistently. 

Here is a demonstration of the fully trained agent playing the game:

![Trained Flappy Bird Agent](gameplay.gif)

*Note: The agent relies purely on its neural network (Exploitation) during this test phase, with the exploration rate (epsilon) decayed to its minimum.*

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
You can adjust the training behavior by editing `parameters.yaml`. Key parameters include:
- `epsilon_decay`: Controls how quickly the agent stops taking random actions.
- `alpha`: The learning rate for the neural network optimizer.
- `gamma`: The discount factor for future rewards.
- `mini_batch_size`: How many memories to sample during each optimization step.
