"""
Holds actual A2C algorithm, implemented using RNN networks.
Much faster with CUDA on GPU's.
With help from https://medium.com/deeplearningmadeeasy/advantage-actor-critic-continuous-case-implementation-f55ce5da6b4c and Chatgpt. 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import tqdm
import warnings

# Disable the warnings from the 'get_namespace_view' call
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


#Make RNN deterministic on CUDA (i.e, same inputs = same outputs each time, will train faster.)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the A2C actor-critic network
class Actor(nn.Module):
    """        Create 'actor' RNN, which will try figure out optimal action for each input.
    """
    def __init__(self, state_dim, action_dim, num_layers=4, hidden_dim=64):
        """

        Parameters
        ----------
        state_dim : int
            Dimensions of input. Done automatically 
        action_dim : int
            Dimensions of output. Double of input (as each dim needs a mean + std to predict)
        num_layers : int
            number of RNN layers in network. The default is 4.
        hidden_dim : int, optional
            Hidden layers in RNN. The default is 64.

        Returns
        -------
        None.

        """
        super(Actor, self).__init__()
        self.RNN = nn.RNN(state_dim, hidden_dim, num_layers, batch_first=True)
        self.Linear = nn.Linear(
            hidden_dim, action_dim * 2
        )  # Double outputs as need a mean and standard deviation to sample from distribution to generate action.

    def forward(self, state):
        '''
        Actually compute mean and standard deviation for given input.
        Input a list of past states.

        Parameters
        ----------
        state : 2D array. [[x,y], [x,y]]
            History of past states. Limited by batch_size.

        Returns
        -------
        action_mean : Array
            Returns means and standard deviations for x and y. Only returns for next (predicted) state.

        '''
        outputs, hidden = self.RNN(state)
        action_mean = self.Linear(
            outputs[-1]
        )  # Only need to pass through final state, hidden layer has been updated.
        return action_mean

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)


class Critic(nn.Module):
    """Create Critic RNN, which will estimate how good the actor is. Only returns one output value per state."""
    def __init__(self, state_dim, num_layers=2, hidden_dim=64):
        """
        No action dim (value should be hardcoded as 1)        
        
        Parameters
        ----------
        state_dim : int
            Dimensions of input. Done automatically 

        num_layers : int
            number of RNN layers in network. The default is 4.
        hidden_dim : int, optional
            Hidden layers in RNN. The default is 64.

        Returns
        -------
        None.

        """
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim

        self.RNN = nn.RNN(state_dim, hidden_dim, num_layers, batch_first=True)
        self.Linear = nn.Linear(hidden_dim, 1)  # Value function is one output

    def forward(self, state):
        '''
        Outputs one value to judge how good actor is, 

        Parameters
        ----------
        state : 2D array. [[x,y], [x,y]]
            History of past states. Limited by batch_size.

        Returns
        -------
        critic_value: Float.

        '''
        outputs, hidden = self.RNN(state)
        critic_value = self.Linear(outputs)
        return critic_value

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)


# A2C agent
class A2CAgent:
    """Agent actually converts inputs to outputs + learns"""
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.0001,
        gamma=0.99,
        grad_clip=0.8,
        output_scaling=1.0,
        entropy_coeff=0.1,  # Greater than 0
        hidden_dim=64,
        num_layers=2,
    ):
        """
        

        Parameters
        ----------
        state_dim : int
            Dimensions of input. 2D = 2
        action_dim : int
            Dimensions of output. 2D = 2
        lr : float, optional
            Affects speed of convergence. The default is 0.0001.
        gamma : float, optional
            Higher means takes into account future rewards more. The default is 0.99.
        grad_clip : float, optional
            Max gradient. The default is 0.8.
        output_scaling : float, optional
            Will limit output of network between +/- of value. The default is 1.0.
        entropy_coeff : Float, optional
            Higher increases exploration. The default is 0.1.
        num_layers : int
            number of RNN layers in network. The default is 4.
        hidden_dim : int, optional
            Hidden layers in RNN. The default is 64.

        Returns
        -------
        None.

        """
        #Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Create Neural networks and optimisers for actor and critic
        self.actor = Actor(state_dim, action_dim, num_layers, hidden_dim).to(
            self.device
        )
        self.critic = Critic(state_dim, num_layers, hidden_dim).to(self.device)
        self.actoroptimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.criticoptimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.output_scaling = output_scaling
        self.rewards_history = []
        self.entropy_coeff = entropy_coeff

    def select_action(self, state, env):
        '''
        

        Parameters
        ----------
        state : numpy array [[x,y], [x,y]]
            History of past states.
        env : environment object.
            

        Returns
        -------
        action : numpy array [x, y]
            Predicted next step.
        action_entropy : float
            Used for entropy.

        '''


        # Convert state to torch tensor

        #If initial state, need special measures.
        check = state == env.start
        if check.all() == True:
            state = torch.unsqueeze(torch.FloatTensor(state).to(self.device), 0)
        else:
            state = torch.FloatTensor(state).to(self.device)

        # Get the action means and stds from the policy network
        action_mean = self.actor(state)  # _ is the hidden state

        # Sample an action from a Gaussian distribution with action_mean as the mean
        #First 2 are means, second 2 are stds
        action_dist = torch.distributions.Normal(
            action_mean[:2], torch.abs(action_mean[2:])
        )  # actor output has 2 means, 2 standard deviations

        action = action_dist.sample()
        action_entropy = action_dist.entropy()

        # Converts to a numpy array - removes the gradient in the tensor first.
        action = action.detach().cpu().numpy()
        return action, action_entropy

    def compute_returns(self, rewards, next_state_values, dones):
        """ Part of A2C algorithm, used to optimise rewards - checks if critic is doing good job."""
        # Compute the expected returns (advantages) for each time step
        returns = []
        R = next_state_values[-1]
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns.insert(0, R)
        return returns

    def train(self, buffer):
        """
        Runs once per episode. Trains agent - this is where learning happens

        Parameters
        ----------
        buffer : Buffer object
            Holds historical info about episode

        Returns
        -------
        loss: float
            Loss, used for graphs.

        """

        #Get required inputs out of buffer.
        # Convert inputs to tensors

        states = torch.FloatTensor(buffer.states).to(self.device)
        actions = torch.FloatTensor(buffer.actions).to(self.device)
        rewards = torch.FloatTensor(buffer.rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(buffer.next_states).to(self.device)
        dones = torch.FloatTensor(buffer.dones).unsqueeze(1).to(self.device)
        entropies = buffer.entropies
        entropies = torch.stack(entropies)
        average_entropy = torch.mean(entropies)  # Find average entropy

        # Compute the action mean and state values for the current states
        action_mean = self.actor(states)
        state_values = self.critic(states)

        # Compute the log probabilities of the actions
        action_log_probs = self._log_prob(actions, action_mean)

        # Compute the action mean and state values for the next states
        next_state_values = self.critic(next_states)

        # Compute the expected returns (advantages) for each time step
        returns = self.compute_returns(rewards, next_state_values, dones)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

        # Compute the advantage
        advantages = returns - state_values

        # Compute the actor and critic losses
        actor_loss = (
            -(action_log_probs * advantages.detach()).mean()
            - self.entropy_coeff * average_entropy
        )
        critic_loss = F.smooth_l1_loss(state_values, returns.detach())

        # Total loss
        loss = actor_loss + critic_loss

        # Backpropagation and optimization with gradient clipping
        self.actoroptimizer.zero_grad()
        self.criticoptimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.actoroptimizer.step()
        self.criticoptimizer.step()

        return loss.item()

    def _log_prob(self, x, mean):
        # Compute the log probability of x given a Gaussian distribution with mean 'mean'
        normal_distribution = torch.distributions.Normal(
            mean[:2], abs(mean[2:])
        )  # Abs so std is positive
        log_prob = normal_distribution.log_prob(x)
        return log_prob

    def plot_rewards_history(self):
        plt.plot(self.rewards_history)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Rewards History")
        plt.show()


class EpisodeBuffer:
    """Holds historical information about episode."""
    def __init__(self):
        #Creates empty lists
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.entropies = []

    def add(self, state, action, reward, next_state, done, entropy):
        """
        

        Parameters
        ----------
        state : numpy array [x, y]
        action : numpy array [x, y]
            Agent's output for input state
        reward : float
        next_state : numpy array [x, y]
            output for action (after going through model)
        done : bool
            Tells if complete
        entropy : float
            Required to calculate mean entropy.

        Returns
        -------
        None.

        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.entropies.append(entropy)

    def clear(self):
        """Used to reset lists"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.entropies.clear()




def pretrain_actor(agent, pretrain_data, env, pretrain_epochs=50, batch_size=50):
    """
    Trajectory that you want to learn from.
    This doesn't work that well - probably buggy.

    Parameters
    ----------
    agent : agent object 
    pretrain_data : Tuple of ideal inputs and outputs (inputs, outputs)
        inputs and outputs should be arrays
    env : environment object.
    pretrain_epochs : int, optional
        Number of times to train. The default is 50.
    batch_size : int, optional
        As before. The default is 50.

    Returns
    -------
    None.

    """

    pretrain_states, pretrain_actions = pretrain_data



    #pretrain states is full history of input states, pretrain actions is full history of input actions.
    #RNN, so convert to trajectories

    #pretrain_states = torch.FloatTensor(pretrain_states).to(agent.device)
    #pretrain_actions = torch.FloatTensor(pretrain_actions).to(agent.device)

    model = agent.actor

    criterion = nn.MSELoss()
    optimizeract = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(pretrain_epochs):
        model.train()
        total_loss = 0

        for i in range(len(pretrain_states)):
            optimizeract.zero_grad()

            input_seq = pretrain_states[:i+1]
            target_seq = pretrain_actions[:i+1]

            if input_seq.shape[0] == 1:
                input_seq = input_seq[0]
                target_seq = target_seq[0]

            if i > batch_size:
                input_seq = input_seq[-batch_size:]
                target_seq = target_seq[-batch_size:]

            output, _ = agent.select_action(input_seq, env)

            output = torch.FloatTensor(output).to(agent.device)
            target_seq = torch.FloatTensor(target_seq).to(agent.device)
            output.requires_grad = True
            target_seq.requires_grad = True

            loss = criterion(output, target_seq)
            loss.backward()
            optimizeract.step()

            total_loss += loss.item()


    print(
        f"Pretrain Epoch {epoch+1}/{pretrain_epochs}, Loss: {total_loss}"
    )
