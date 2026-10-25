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

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
    
    def run(self, is_training = True, render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)
    
        num_states = env.observation_space.shape[0] #input dime
        num_actions = env.action_space.n    #output dim
    
        policy_dqn = DQN(num_states, num_actions).to(device)
    
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

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
            epsilon = 0
    
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype = torch.float, device=device)

            step = 0
            
            episode_reward = 0
            terminated = False
            
            while (not terminated and episode_reward < self.reward_threshold):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()    #EXPLORE
                else:
                    with torch.inference_mode():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()    #EXPLOIT
                    
                action = torch.tensor(action, dtype = torch.long, device=device)
        
                # Processing:
                next_state, reward, terminated, _, _ = env.step(action.item())
                
                next_state = torch.tensor(next_state, dtype = torch.float, device=device)
                
                episode_reward += reward
                
                reward = torch.tensor(reward, dtype = torch.float, device=device)
        
                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    step += 1
                    target_copy_step += 1
    
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
                print(f"Ep: {episode+1} \tTotal Reward: {episode_reward:.10f} \tEpsilon: {epsilon:.5f}")

                #Epsilon Decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if episode_reward > best_rewards:
                    log_msg = f"Best Reward: {episode_reward} for Ep:{episode+1}"

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_rewards = episode_reward
    
        #env.close()     #Manual Stop

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        #get experience  =>  batch training
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        # Calculate target Q-value | If termination=True ==> Zero
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # Calculate y_pred i.e Q-value from curr policy
        current_q = policy_dqn(states).gather(dim=1, index = actions.unsqueeze(dim=1)).squeeze()

        #loss
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
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