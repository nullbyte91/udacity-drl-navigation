import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from dueling_model import DuelingQNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 4.8e-4             # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_DOUBLE_DQN = True
USE_PRIORITIZED_REPLAY_BUFFER = False
USE_DUELING_NETWORK = True

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, lr_decay=0.9999):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        if USE_DUELING_NETWORK:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed, 
                                                [128, 32], [64, 32]).to(device)

            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed, 
                                                [128, 32], [64, 32]).to(device)
            self.qnetwork_target.eval()
            
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, 
                                            seed, fc1_units= 128, fc2_units=32).to(device)

            self.qnetwork_target = QNetwork(state_size, action_size, 
                                            seed, fc1_units= 128, fc2_units=32).to(device)
            self.qnetwork_target.eval()
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)

        # Replay memory
        if USE_PRIORITIZED_REPLAY_BUFFER:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device, 
                            alpha=0.6, beta=0.4, beta_scheduler=1.0)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, w = experiences

        if USE_DOUBLE_DQN:
            self.qnetwork_local.eval()
            Q_local = self.qnetwork_local(next_states)
            greedy_actions = torch.argmax(Q_local, axis=1).unsqueeze(1)
            self.qnetwork_local.train()

            Q_targets_next = self.qnetwork_target(next_states)
            Q_targets_next = Q_targets_next.gather(1, greedy_actions)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        else:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            
            Q_targets_next = Q_targets_next.max(dim=1, keepdim=True)[0]

            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        if USE_PRIORITIZED_REPLAY_BUFFER:
            Q_targets.sub_(Q_expected)
            Q_targets.squeeze_()
            Q_targets.pow_(2)

            with torch.no_grad():
                td_error = Q_targets.detach()
                td_error.pow_(0.5)
                self.memory.update_priorities(td_error)

            Q_targets.mul_(w)
            loss = Q_targets.mean()
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        # Mback-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, None)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """Fixed-size prioritized buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, device, alpha=0., beta=1., beta_scheduler=1.0):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): determines how much prioritization is used; α = 0 corresponding to the uniform case
            beta (float): amount of importance-sampling correction; β = 1 fully compensates for the non-uniform probabilities
        """
        
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = np.random.seed(seed)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_scheduler = beta_scheduler

        # Create a Numpy Array to store tuples of experience
        self.memory = np.empty(buffer_size, dtype=[
            ("state", np.ndarray),
            ("action", np.int),
            ("reward", np.float),
            ("next_state", np.ndarray),
            ("done", np.bool),
            ('prob', np.double)])
        # Variable to control the memory buffer as being a circular list
        self.memory_idx_ctrl = 0
        
        # Variable to control the selected samples
        self.memory_samples_idx = np.empty(batch_size)
        # Numpy Array to store selected samples
        # Those samples could be controlled only by the index,
        # however keeping an allocated space in memory improves performance.
        # (Here we have a tradeoff between memory space and cumputacional processing)
        self.memory_samples = np.empty(batch_size, dtype=type(self.memory))

        # Each new experience is added to the memory with
        # the maximum probability of being choosen
        self.max_prob = 0.0001
        
        # Value to a non-zero probability
        self.nonzero_probability = 0.00001
        
        # Numpy Arrays to store probabilities and weights
        # (tradeoff between memory space and cumputacional processing)
        self.p = np.empty(buffer_size, dtype=np.double)
        self.w = np.empty(buffer_size, dtype=np.double)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        # Add the experienced parameters to the memory
        self.memory[self.memory_idx_ctrl]['state'] = state
        self.memory[self.memory_idx_ctrl]['action'] = action
        self.memory[self.memory_idx_ctrl]['reward'] = reward
        self.memory[self.memory_idx_ctrl]['next_state'] = next_state
        self.memory[self.memory_idx_ctrl]['done'] = done
        self.memory[self.memory_idx_ctrl]['prob'] = self.max_prob
        
        # Control memory as a circular list
        self.memory_idx_ctrl = (self.memory_idx_ctrl + 1) % self.buffer_size
    
    def sample(self):
        """Sample a batch of prioritized experiences from memory."""
        
        # Normalize the probability of being chosen for each one of the memory registers
        np.divide(self.memory['prob'], self.memory['prob'].sum(), out=self.p)
        # Choose "batch_size" sample index following the defined probability
        self.memory_samples_idx = np.random.choice(self.buffer_size, self.batch_size, replace=False, p=self.p)
        # Get the samples from memory
        self.memory_samples = self.memory[self.memory_samples_idx]
        
        # Compute importance-sampling weights for each one of the memory registers
        # w = ((N * P) ^ -β) / max(w)
        np.multiply(self.memory['prob'], self.buffer_size, out=self.w)
        np.power(self.w, -self.beta, out=self.w, where=self.w!=0) # condition to avoid division by zero
        np.divide(self.w, self.w.max(), out=self.w) # normalize the weights
        self.beta = min(1, self.beta*self.beta_scheduler)

        # Split data into new variables
        states = torch.from_numpy(np.vstack(self.memory_samples['state'])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(self.memory_samples['action'])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(self.memory_samples['reward'])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(self.memory_samples['next_state'])).float().to(self.device)
        dones = torch.from_numpy(np.vstack(self.memory_samples['done'])).float().to(self.device)
        weights = torch.from_numpy(self.w[self.memory_samples_idx]).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones, weights)
    
    def update_priorities(self, td_error):
        # Balance the prioritization using the alpha value
        td_error.pow_(self.alpha)

        # Guarantee a non-zero probability
        td_error.add_(self.nonzero_probability)
        
        # Update the probabilities in memory
        self.memory_samples['prob'] = td_error.cpu()
        self.memory[self.memory_samples_idx] = self.memory_samples
        
        # Update the maximum probability value
        self.max_prob = self.memory['prob'].max()
        
       
    def __len__(self):
        """Return the current size of internal memory."""
        return self.buffer_size if self.memory_idx_ctrl // self.buffer_size > 0 else self.memory_idx_ctrl