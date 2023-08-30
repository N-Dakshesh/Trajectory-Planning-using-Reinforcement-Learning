# -*- coding: utf-8 -*-
"""
This is a Reinforcement Learning Environment. It converts inputs to outputs, 
and calculates a reward. 
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import data_gen_toolsv3 as dgt  

a= 0

def path_func(x, y, sigma=0.05, y0=0, xi=0, xf=0.5):
    # x, y = xy
    # Define to return - 1 at peak.

    output = np.array([])

    if x <= xi:
        pwr = ((x - xi) * (x - xi) + (y - y0) * (y - y0)) / (2 * sigma * sigma)
        out = np.exp(-pwr)
        output = np.append(output, out)
    elif x > xf:
        pwr = ((x - xf) * (x - xf) + (y - y0) * (y - y0)) / (2 * sigma * sigma)
        out = np.exp(-pwr)
        output = np.append(output, out)
    else:
        pwr = (y - y0) * (y - y0) / (2 * sigma * sigma)
        out = np.exp(-pwr)
        output = np.append(output, out)

    return output + x


class CustomEnv(gym.Env):

    def __init__(
        self,
        timestep=0.1,
        maxv=1,
        target=np.array([0.5, 0.0]),
        start = np.array([0.0, 0.0]),
        tolerance=0.1,
        end_reward=0.2,
        cuttoff_time=20,
        reward_func = a,
        mode = 'constant'
    ):
        """
        
        Note: input space is hardcoded to be between +/- 1, so rescale anything larger.
        If you want to change it, you can in the code, but rescaling is recommended.


        Parameters
        ----------
        timestep : Float, optional
            Timestep incremented by. The default is 0.1.
        maxv : Float, optional
            Velocity limit. If inputs with greater velocity given, limited to this velocity. The default is 1.
        target : 2 element array, optional
            End point. The default is np.array([0.5, 0.0]).
        start : 2 element array, optional
            Start Point. The default is np.array([0.0, 0.0]).
        tolerance : Float, optional
            Distance from the endpoint to count as complete. The default is 0.1.
        end_reward : Float, optional
            Reward for reaching target. The default is 0.2.
        cuttoff_time : Float, optional
            Max time before failure. The default is 20.
        reward_func : Function, either path_func (in this file) or func generated from path gen, optional
            Used to calculate rewards. The default is a.
        mode : String, optional
            constant, accelerate or constant_experimental. Chooses which model to use. The default is 'constant'.

        Raises
        ------
        Exception
            If reward function isn't input

        Returns
        -------
        None.

        """

        super().__init__()


        if reward_func == a:
            raise Exception("Put in a valid reward function")

        # Define the action space.
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))

        # Define the observation space. (Doesn't really matter)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(2,))
        self.state = start  # Input state
        self.output = start #Initial output = input

        self.all_inputs = start
        self.all_outputs = start
        self.target = target  # Need to be able to define manually.

        self.time = 0.0
        self.all_times = np.array([0.0]) #initial time is 0
        self.timestep = timestep
        self.reward = 0

        self.maxv = maxv  # Maximum velocity constraint
        self.current_step = 0
        self.state_dim = 2 #2D
        self.action_dim = 2

        self.tolerance = tolerance
        self.cuttoff_time = cuttoff_time
        self.end_reward = end_reward

        self.reward_func = reward_func
        self.mode = mode
        self.start = start #To remember the initial point, as state gets updated

    def reset(self):
        """
        Resets all parameters to the beginning.

        Returns
        -------
        2 element array, [x,y]
            Initial State - complete reset.

        """
        # Reset the environment to a random state.
        self.state = self.start  # Return to initial conditions
        self.all_inputs = self.start
        self.all_outputs = self.start
        self.time = 0.0
        self.all_times = np.array([0.0])
        self.current_step = 0
        self.reward = 0

        return self.state

    def step(
        self, action
    ):
        """
        Takes step forward for input [x, y], and calculates a reward.

        Parameters
        ----------
        action : Numpy array [x, y]
            Next step, predicted by agent

        Returns
        -------
        self.state: Numpy array [x, y] 
            output state (after going through model)
        self.reward: Float
            Reward. See presentation to understand reward calculation + extra reward if reached goal.
        done : Bool (True/False)
            Tells you if reached goal
        dict: None
            n/a

        """

        # action will be taken at fixed timesteps

        old_value = self.reward_func(self.state[0], self.state[1]) #Calculate current point's value at reward func

        #Calculate and constrain velocity
        change = action - self.state
        v = change / self.timestep
        v_mag = np.linalg.norm(v)

        if v_mag > self.maxv:
            # Need to make sure step is in the same direction as v.
            unit_v = v / v_mag
            action = (
                self.state + unit_v * self.maxv * self.timestep
            )  # Velocity unit velocity, multiplied by magnitude and time



        self.time = self.time + self.timestep
        self.state = action  #Move forward

        self.all_times = np.append(
            self.all_times, self.time
        )  # As have updated self.time.

        self.all_inputs = np.vstack(
            [self.all_inputs, self.state]
        )  # As have updated self.state

        #Converting inputs to outputs using Transfer function.
        #Independent in x and y, so convert separately.
        ux = self.all_inputs[:, 0]
        uy = self.all_inputs[:, 1]
        times = np.linspace(0, self.time, len(self.all_inputs))
        x = dgt.outputs(times, ux, self.mode)
        y = dgt.outputs(times, uy, self.mode)
        self.output = np.array([x[-1], y[-1]])

        # Check if the episode is done (target reached)

        distance = abs(np.linalg.norm(self.output - self.target))
        done = distance < self.tolerance  # i.e close/max steps.

        # Calculate the reward based on the final time if the episode is done
        if done == True:
            reward = self.end_reward
            self.reward = reward + 1/(1+self.time)
            print("done")
        #If taking too long, cut early with no reward
        elif self.time > self.cuttoff_time:
            reward = 0
            done = True
            self.reward = reward
        else:
            #Use value function to give reward dependent on position
            value = self.reward_func(self.state[0], self.state[1])
            reward = value - old_value
            #Seems to work better if *100. You can experiment
            self.reward = reward*100

        self.all_outputs = np.transpose(np.array([x, y]))
        self.output = np.array([x[-1], y[-1]])
        self.render()

        # Return the next state, reward, done flag, and additional info (empty dictionary)
        #-2 FROM ALL REWARDS! Seems to make it work better.
        return self.state, self.reward - 2, done, {}

    def render(self, mode="human"):
        """
        Render function is standard for environments. Uncomment code if you actually want
        to print every step (which you don't)

        When functional, will print current time, position and reward. 

        """
        # print("Time is", self.time, "Postion is", self.output, "Reward is", self.reward)
        1 == 1



