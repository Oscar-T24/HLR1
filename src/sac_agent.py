"""Soft Actor-Critic agent and replay buffer."""

from __future__ import annotations

import numpy as np
import torch as T
import torch.nn.functional as F

from .networks import ActorNetwork, CriticNetwork, ValueNetwork


class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        i = self.mem_cntr % self.mem_size
        self.state_memory[i] = state
        self.new_state_memory[i] = state_
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.terminal_memory[i] = bool(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.new_state_memory[batch],
            self.terminal_memory[batch],
        )


class SACAgent:
    def __init__(
        self,
        alpha=3e-4,
        beta=3e-4,
        input_dims=None,
        gamma=0.99,
        n_actions=6,
        max_size=1_000_000,
        tau=0.005,
        batch_size=256,
        reward_scale=1.0,
        max_action=1.0,
        layer1_size=256,
        layer2_size=256,
        chkpt_dir="./checkpoints",
    ):
        if input_dims is None:
            input_dims = [19]
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale
        self.max_action = max_action

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.actor = ActorNetwork(alpha, input_dims, max_action=max_action, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions, name="actor", chkpt_dir=chkpt_dir)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, fc1_dims=layer1_size, fc2_dims=layer2_size, name="critic_1", chkpt_dir=chkpt_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, fc1_dims=layer1_size, fc2_dims=layer2_size, name="critic_2", chkpt_dir=chkpt_dir)
        self.value = ValueNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, name="value", chkpt_dir=chkpt_dir)
        self.target_value = ValueNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, name="target_value", chkpt_dir=chkpt_dir)
        self.update_network_parameters(tau=1.0)

    def choose_action(self, observation, evaluate=False):
        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)
        if evaluate:
            mu, _ = self.actor.forward(state)
            actions = T.tanh(mu) * self.max_action
        else:
            actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        target_value_state_dict = dict(self.target_value.named_parameters())
        value_state_dict = dict(self.value.named_parameters())
        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1 - tau) * target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        action = T.tensor(action, dtype=T.float32).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)
        new_state = T.tensor(new_state, dtype=T.float32).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(new_state).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        critic_value = T.min(self.critic_1(state, actions), self.critic_2(state, actions)).view(-1)

        self.value.optimizer.zero_grad()
        value_loss = 0.5 * F.mse_loss(value, (critic_value - log_probs).detach())
        value_loss.backward()
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        critic_value = T.min(self.critic_1(state, actions), self.critic_2(state, actions)).view(-1)
        actor_loss = T.mean(log_probs - critic_value)

        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_.detach()
        q1_old_policy = self.critic_1(state, action).view(-1)
        q2_old_policy = self.critic_2(state, action).view(-1)
        critic_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat) + 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

