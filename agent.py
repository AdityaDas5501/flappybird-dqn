import numpy as np
import gymnasium
import flappy_bird_gymnasium
import itertools
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using {device} device")
  
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set
        
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.env_id = params["env_id"]
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        self.epsilon = self.epsilon_init

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
    
    def run(self, is_training = True, render=False):
        env = gymnasium.make(self.env_id, render_mode="human" if render else None)
    
        num_states = env.observation_space.shape[0] #input dime
        num_actions = env.action_space.n    #output dim
    
        policy_dqn = DQN(num_states, num_actions).to(device)
    
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            target_dqn = DQN(num_states, num_actions).to(device)
            #copy wt. and bias values from policy to target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            target_copy_step = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr = self.alpha)

            best_rewards = float("-inf")

        else:
            # Load Best Model with exception handling
            try:
                # Try loading it as it was saved
                policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
                print("Model loaded successfully on the default device.")
            except RuntimeError:
                # Catch the CUDA mismatch error and force it to the available device (CPU)
                print(f"GPU not found. Falling back to {device}...")
                policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))

            policy_dqn.eval()
            self.epsilon = 0
    
        for episode in itertools.count():
            state, _ = env.reset()

            step = 0
            
            episode_reward = 0
            terminated = False
            
            while (not terminated and episode_reward < self.reward_threshold):
                if is_training and random.random() < self.epsilon:
                    action = env.action_space.sample()    #EXPLORE
                else:
                    with torch.inference_mode():
                        state_tensor = torch.tensor(state, dtype = torch.float, device=device)
                        action = policy_dqn(state_tensor.unsqueeze(dim=0)).squeeze().argmax().item()    #EXPLOIT
        
                # Processing:
                next_state, reward, terminated, _, _ = env.step(action)

                if terminated:
                    modified_reward = -2.0
                elif reward > 0:
                    modified_reward = 5.0
                else:
                    modified_reward = 0.02
                
                episode_reward += reward
                total_modified_score += modified_reward
        
                if is_training:
                    memory.append((state, action, next_state, modified_reward, terminated))
                    step += 1
                    target_copy_step += 1

                    #Epsilon Decay
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
                state = next_state

                if is_training and len(memory) > self.mini_batch_size:
                    #get sample
                    mini_batch = memory.sample(self.mini_batch_size)
    
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    #sync the network
                    if target_copy_step > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        target_copy_step = 0

            if is_training:
                print(f"Ep: {episode+1} \tTotal Raw Reward: {episode_reward:.10f} \tModified Reward: {total_modified_score:.5f} \tEpsilon: {self.epsilon:.5f}")

                if episode_reward > best_rewards:
                    log_msg = f"Best Raw Reward: {episode_reward::.5f} (Modified Reward: {total_modified_score:.5f}) for Ep:{episode+1}"

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_rewards = episode_reward
    
        #env.close()     #Manual Stop

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        #get experience  =>  batch training
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)

        # Calculate target Q-value | If termination=True ==> Zero
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # Calculate y_pred i.e Q-value from curr policy
        current_q = policy_dqn(states).gather(dim=1, index = actions.unsqueeze(dim=1)).squeeze(1)

        #loss
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser (description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)