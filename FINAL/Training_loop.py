"""Training loop for the neural net. Works with A2C_RNN_work, which is the A2C algorithm with a RNN."""

import os
from datetime import datetime
import A2C_main as net
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import tqdm
import torch
import data_gen_toolsv3 as dgt


# %%


def initHidden(hidden_dim=64):
    return torch.zeros(hidden_dim)


def training(
    lr,
    entropy_coeff,
    hidden_dim,
    num_layers,
    gamma,
    end_reward,
    cuttoff_time,
    reward_func,
    start,
    target,
    batch_size=50,
    episode_no = 100,
    entry_point = "EnvA2C_v4:CustomEnv",
    mode = 'constant',
    disable_graphs = True,
    pretrain = False


):
    """
    

    Parameters
    ----------
    lr : float
        Affects speed of convergence. 
    gamma : float
        Higher means takes into account future rewards more. T
    entropy_coeff : Float
        Higher increases exploration. 
    num_layers : int
        number of RNN layers in network. 
    hidden_dim : int, optional
        Hidden layers in RNN. The default is 64.
    end_reward : float
        Reward for reaching goal.
    cuttoff_time : float
        max time before force stop.
    reward_func : path_gen reward function.
        Use to define path.
    start : numpy array [x, y]
        Initial point
    target : numpy array [x, y]
        Final point.
    batch_size : int, optional
        Number of past timesteps RNN should remember - set depending on length of transfer function and timesteps. The default is 50.
    episode_no : int, optional
        Number of times to learn. More is normally better, but could break. The default is 100.
    entry_point : string, optional
        Location of environment + name of environment inside. The default is "EnvA2C_v4:CustomEnv".
    mode : string, constant, accelerate or constant_experimental
        Chooses which transfer function to use. The default is 'constant'. constant_experimental combines both - but only valid for constant velocity with discontinuties.
    disable_graphs : bool, optional
        Do you want graphs. The default is True.
    pretrain : bool, optional
        Doesn't really work so don't recommend. The default is False.

    Returns
    -------
    final_time: float
        Returns final time of best run (i.e. min time)
    max_data : 2D numpy array [[x, y], [x, y]]
        Inputs for best episode.
    max_data_output : 2D numpy array [[x, y], [x, y]]
        Outputs for best episode.
    every_loss : numpy array.
        History of losses.

    """
    # Say max velocity is 1, and target is one.
    #Fix timestep to 0.1
    timestep = 0.1
    gym.envs.register("Scanner/-3", entry_point= entry_point)

    #Make environment and agent.
    env = gym.make(
        "Scanner/-3",
        maxv=0.1,
        timestep=timestep,
        cuttoff_time=cuttoff_time,
        end_reward=end_reward,
        reward_func = reward_func,
        target = target,
        mode = mode,
        start = start

    )
    agent = net.A2CAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=lr,
        entropy_coeff=entropy_coeff,
        gamma=gamma,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    every_input = []
    every_output = []
    every_loss = []

    #Don't use until you fix it
    if pretrain == True:

        # Load the initial training data
        # pretrain_states and pretrain_actions should be lists or arrays of state and corresponding action data
        times = np.arange(0, 8, 0.1) #Not time optimal, but reaches the goal, so good starting
        x_vals = times*0.5*(1/8) #i.e. at 8 seconds, reaches 0.5
        y_vals = np.zeros(len(x_vals))
    
        pstates = np.array([x_vals, y_vals])
        pstates = pstates.transpose()
        pactionsx = dgt.outputs(times, x_vals, mode)
        pactionsy = dgt.outputs(times, y_vals, mode)
        pactions = np.array([pactionsx, pactionsy])
        pactions = pactions.transpose()
    
    
        # Pretrain the actor network
        pretrain_data = (pstates, pactions)
        net.pretrain_actor(agent, pretrain_data, env = env)
        print('Pretraining Complete')


    #Title for graphs - so you
    timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
    folder_name = f"Run_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    num_episodes = episode_no
    i = 0


    #tqdm creates a loading bar.
    for episode in tqdm.tqdm(
        range(num_episodes), desc="Episodes", position=0, leave=True
    ):
        #initial parameters
        state = env.reset()
        done = False
        total_reward = 0
        buffer = net.EpisodeBuffer()

        # Need
        j = 0


        while not done:
            all_states = env.all_inputs

            #Tests if initial step (if initial, state = [x, y], so len gives 2 when want 1)
            if all_states.ndim != 1:

                #Limit inputs to batch size.
                if len(all_states) > batch_size:
                    all_states = all_states[-batch_size:]

            action, action_entropy = agent.select_action(all_states, env)

            #Ensures 1D array.
            if action.ndim != 1:
                action = action[-1]

            next_state, reward, done, _ = env.step(action)
            reward = float(reward)
            total_reward += reward

            #Store in memory
            buffer.add(state, action, reward, next_state, done, action_entropy)
            state = next_state
            j = j + 1

        if done == True:
            #Add to list of historical inputs and outputs.
            every_input.append(env.all_inputs)
            every_output.append(env.all_outputs)

        i = i + 1
        if buffer.states:  # Only train if the buffer is not empty
            loss = agent.train(buffer)
            every_loss.append(loss)

        # Train the agent after each episode

        # Keep track of rewards history
        agent.rewards_history.append(total_reward)
        buffer.clear()

        # print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

    # print("Training Complete!")

    a = agent.rewards_history

    if disable_graphs == True:
        max_arg = np.argmax(a)
        max_data = every_input[max_arg]
        max_data_output = every_output[max_arg]
        times = np.arange(0, len(max_data))*env.timestep

    else:
        max_arg = np.argmax(a)
        plt.figure(figsize=[8, 6])
        plt.plot(a, label="Max Reward is {}".format(a[max_arg]))
        plt.title("Episode Rewards History_{}".format(folder_name))
        plt.grid()
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.legend()
        plt.plot()
        plt.show()
    
        plt.figure(figsize=[8, 6])
    
        max_data = every_input[max_arg]
        times = np.arange(0, len(max_data)) * timestep
        plt.plot(times, max_data[:, 0], label="x")
        plt.plot(times, max_data[:, 1], label="y")
        plt.title("Best Trajectory Inputs_{}".format(folder_name))
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.plot()
        plt.show()

    
        plt.figure(figsize=[8, 6])
        max_data_output = every_output[max_arg]
        times = np.arange(0, len(max_data_output)) * timestep
        plt.plot(times, max_data_output[:, 0], label="x")
        plt.plot(times, max_data_output[:, 1], label="y")
        plt.title("Best Trajectory Outputs_{}".format(folder_name))
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.plot()
        plt.show()


    return times[-1], max_data, max_data_output, every_loss
