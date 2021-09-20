import gym
import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple
import numpy as np
from collections import deque
from copy import copy

from ..data.buffers import Experience, ReplayBuffer


class Agent:
    """Base Agent class handling the interaction with the environment.
    >>> env = gym.make("CartPole-v1")
    >>> buffer = ReplayBuffer(10)
    >>> Agent(env, buffer)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.Agent object at ...>
    """

    def __init__(
        self, 
        env: gym.Env, 
        replay_buffer: ReplayBuffer,
        stack_length: int = 16,
    ) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.stack_length = stack_length

        first_state = self.env.reset()
        self.state = deque([first_state for _ in range(stack_length)], maxlen=stack_length)
        self.next_state = deque([first_state for _ in range(stack_length)], maxlen=stack_length)
        #self.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        single_state = self.env.reset()
        self.state = deque([single_state for _ in range(self.stack_length)], maxlen=self.stack_length)
        self.next_state = deque([single_state for _ in range(self.stack_length)], maxlen=self.stack_length)

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])

            if device not in ["cpu"]:
                state = state.cuda(device)
            # TODO apply softmax probabilities here
            q_values = net(state)
            #_, action = torch.max(q_values, dim=1)
            probabilities = F.softmax(q_values, dim=1) # 2d shape
            action = int(torch.multinomial(probabilities, 1).item())

            #action = int(action.item())

        return action

    def log(self):
        """Log the current state of the env."""
        return {
            **self.replay_buffer.log(),
            'won': self.env.n_won_battles,
            'lost': self.env.n_lost_battles,
            'tied': self.env.n_tied_battles,
            #'total_reward': self.env.total_reward,
            'win_rate': self.env.win_rate,
        }

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = "cpu") -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        self.next_state.append(new_state)

        exp = Experience(list(self.state), action, reward, done, list(self.next_state))

        self.replay_buffer.append(exp)

        self.state.append(new_state)
        if done:
            self.reset()
        return reward, done

