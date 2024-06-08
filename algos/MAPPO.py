import argparse
import time
from typing import List, Union
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
from torch_geometric.utils import add_self_loops, to_dense_batch

from onpolicy.popart import PopArt
from onpolicy.rnn import RNNLayer
from onpolicy.util import init, check, get_clones
from onpolicy.utils.graph_buffer import GraphReplayBuffer
from onpolicy.utils.util import get_grad_norm, huber_loss, mse_loss
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.utils.util import update_linear_schedule
from onpolicy.utils.valuenorm import ValueNorm
from algos.action_out import ACTLayer
from algos.mlp import MLPBase
from torch_geometric.data import DataLoader, Data
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from torch_geometric.nn.pool import TopKPooling


class GR_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks
    to compute actions and value function predictions.

    args: (argparse.Namespace)
        Arguments containing relevant model and policy information.
    obs_space: (gym.Space)
        Observation space.
    cent_obs_space: (gym.Space)
        Value function input space
        (centralized input for MAPPO, decentralized for IPPO).
    node_obs_space: (gym.Space)
        Node observation space
    edge_obs_space: (gym.Space)
        Edge dimension in graphs
    action_space: (gym.Space) a
        Action space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    """

    def __init__(self, obs_dim, num_services=8, svc_adj=None) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = 0.0005
        self.critic_lr = 0.0005
        self.opti_eps = 1e-05
        self.weight_decay = 0

        self.split_batch = False
        self.max_batch_size = 32

        self.actor = GR_Actor(self.device, obs_dim, num_services, svc_adj=svc_adj)
        self.critic = GR_Critic(self.device, obs_dim, svc_adj=svc_adj)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode: int, episodes: int) -> None:
        """
        Decay the actor and critic learning rates.
        episode: (int)
            Current training episode.
        episodes: (int)
            Total number of training episodes.
        """
        update_linear_schedule(
            optimizer=self.actor_optimizer,
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.lr,
        )
        update_linear_schedule(
            optimizer=self.critic_optimizer,
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.critic_lr,
        )

    def get_actions(self, cent_obs, obs, node_obs, adj, agent_id, share_agent_id, rnn_states_actor, rnn_states_critic,
                    masks, available_actions=None, deterministic=False, svc_obs=None,
                    svc_adj=None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute actions and value function predictions for the given inputs.
        cent_obs (np.ndarray):
            Centralized input to the critic.
        obs (np.ndarray):
            Local agent inputs to the actor.
        node_obs (np.ndarray):
            Local agent graph node features to the actor.
        adj (np.ndarray):
            Adjacency matrix for the graph.
        agent_id (np.ndarray):
            Agent id to which observations belong to.
        share_agent_id (np.ndarray):
            Agent id to which cent_observations belong to.
        rnn_states_actor: (np.ndarray)
            If actor is RNN, RNN states for actor.
        rnn_states_critic: (np.ndarray)
            If critic is RNN, RNN states for critic.
        masks: (np.ndarray)
            Denotes points at which RNN states should be reset.
        available_actions: (np.ndarray)
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            Whether the action should be mode of
            distribution or should be sampled.

        :param svc_adj:
        :param svc_obs:
        :return values: (torch.Tensor)
            value function predictions.
        :return actions: (torch.Tensor)
            actions to take.
        :return action_log_probs: (torch.Tensor)
            log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor)
            updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor)
            updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor.forward(obs, node_obs, adj, agent_id, rnn_states_actor,
                                                                         masks, available_actions, deterministic,
                                                                         svc_obs=svc_obs, svc_adj=svc_adj)

        values, rnn_states_critic = self.critic.forward(cent_obs, node_obs, adj, share_agent_id, rnn_states_critic,
                                                        masks, svc_obs=svc_obs, svc_adj=svc_adj)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks, svc_obs=None,
                   svc_adj=None) -> Tensor:
        """
        Get value function predictions.
        cent_obs (np.ndarray):
            centralized input to the critic.
        node_obs (np.ndarray):
            Local agent graph node features to the actor.
        adj (np.ndarray):
            Adjacency matrix for the graph.
        share_agent_id (np.ndarray):
            Agent id to which cent_observations belong to.
        rnn_states_critic: (np.ndarray)
            if critic is RNN, RNN states for critic.
        masks: (np.ndarray)
            denotes points at which RNN states should be reset.

        :param svc_adj:
        :param svc_obs:
        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic.forward(cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks,
                                        svc_obs=svc_obs, svc_adj=svc_adj)
        return values

    def evaluate_actions(self, cent_obs, obs, node_obs, adj, agent_id, share_agent_id, rnn_states_actor,
                         rnn_states_critic, action, masks, available_actions=None, active_masks=None, svc_obs=None,
                         svc_adj=None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get action logprobs / entropy and
        value function predictions for actor update.
        :param svc_adj:
        :param cent_obs: (np.ndarray):
            centralized input to the critic.
        :param obs: (np.ndarray):
            local agent inputs to the actor.
        :param node_obs: (np.ndarray):
            Local agent graph node features to the actor.
        :param adj: (np.ndarray):
            Adjacency matrix for the graph.
        :param agent_id: (np.ndarray):
            Agent id for observations
        :param share_agent_id: (np.ndarray):
            Agent id for shared observations
        :param rnn_states_actor: (np.ndarray)
            if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray)
            if critic is RNN, RNN states for critic.
        :param action: (np.ndarray)
            actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray)
            denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray)
            denotes which actions are available to agent
            (if None, all actions available)
        :param active_masks: (torch.Tensor)
            denotes whether an agent is active or dead.
        :param svc_obs:

        :return values: (torch.Tensor)
            value function predictions.
        :return action_log_probs: (torch.Tensor)
            log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor)
            action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, node_obs, adj, agent_id, rnn_states_actor,
                                                                     action, masks, available_actions, active_masks,
                                                                     svc_obs=svc_obs, svc_adj=svc_adj)

        values, _ = self.critic.forward(cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks,
                                        svc_obs=svc_obs, svc_adj=svc_adj)
        return values, action_log_probs, dist_entropy

    def act(
            self,
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_actor,
            masks,
            available_actions=None,
            deterministic=False,
            svc_obs=None,
            svc_adj=None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute actions using the given inputs.
        obs (np.ndarray):
            local agent inputs to the actor.
        node_obs (np.ndarray):
            Local agent graph node features to the actor.
        adj (np.ndarray):
            Adjacency matrix for the graph.
        agent_id (np.ndarray):
            Agent id for nodes for the graph.
        rnn_states_actor: (np.ndarray)
            if actor is RNN, RNN states for actor.
        masks: (np.ndarray)
            denotes points at which RNN states should be reset.
        available_actions: (np.ndarray)
            denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            whether the action should be mode of
            distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor.forward(obs, node_obs, adj, agent_id, rnn_states_actor, masks,
                                                          available_actions, deterministic,
                                                          svc_obs=svc_obs,
                                                          svc_adj=svc_adj)
        return actions, rnn_states_actor


class GR_MAPPO:
    """
    Trainer class for Graph MAPPO to update policies.
    args: (argparse.Namespace)
        Arguments containing relevant model, policy, and env information.
    policy: (GR_MAPPO_Policy)
        Policy to update.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    """

    def __init__(
            self,
            policy: GR_MAPPOPolicy,
            device=torch.device("cpu"),
    ) -> None:
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = 0.2
        self.ppo_epoch = 5
        self.num_mini_batch = 1
        self.data_chunk_length = 10
        self.value_loss_coef = 1
        self.entropy_coef = 0.01
        self.max_grad_norm = True
        self.huber_delta = 10.0

        self._use_recurrent_policy = True
        self._use_naive_recurrent = False
        self._use_max_grad_norm = True
        self._use_clipped_value_loss = True
        self._use_huber_loss = True
        self._use_popart = False
        self._use_valuenorm = True
        self._use_value_active_masks = True
        self._use_policy_active_masks = True

        assert (
                       self._use_popart and self._use_valuenorm
               ) == False, "self._use_popart and self._use_valuenorm can not be set True simultaneously"

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(
            self,
            values: Tensor,
            value_preds_batch: Tensor,
            return_batch: Tensor,
            active_masks_batch: Tensor,
    ) -> Tensor:
        """
        Calculate value function loss.
        values: (torch.Tensor)
            value function predictions.
        value_preds_batch: (torch.Tensor)
            "old" value  predictions from data batch (used for value clip loss)
        return_batch: (torch.Tensor)
            reward to go returns.
        active_masks_batch: (torch.Tensor)
            denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor)
            value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (
                    self.value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (
                                 value_loss * active_masks_batch
                         ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(
            self, sample: Tuple, update_actor: bool = True
    ):
        """
        Update actor and critic networks.
        sample: (Tuple)
            contains data batch with which to update networks.
        update_actor: (bool)
            whether to update actor network.

        :return value_loss: (torch.Tensor)
            value function loss.
        :return critic_grad_norm: (torch.Tensor)
            gradient norm from critic update.
        ;return policy_loss: (torch.Tensor)
            actor(policy) loss value.
        :return dist_entropy: (torch.Tensor)
            action entropies.
        :return actor_grad_norm: (torch.Tensor)
            gradient norm from actor update.
        :return imp_weights: (torch.Tensor)
            importance sampling weights.
        """
        start_time = time.time()
        (
            share_obs_batch,
            obs_batch,
            svc_obs_batch,
            node_obs_batch,
            adj_batch,
            svc_adj_batch,
            agent_id_batch,
            share_agent_id_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch, obs_batch,
                                                                              node_obs_batch, adj_batch, agent_id_batch,
                                                                              share_agent_id_batch, rnn_states_batch,
                                                                              rnn_states_critic_batch, actions_batch,
                                                                              masks_batch, available_actions_batch,
                                                                              active_masks_batch, svc_obs=svc_obs_batch,
                                                                              svc_adj=svc_adj_batch)
        # actor update
        # print(f'obs: {obs_batch.shape}')
        # st = time.time()
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = (
                torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * adv_targ
        )
        # print(f'Surr1: {surr1.shape} \t Values: {values.shape}')

        if self._use_policy_active_masks:
            policy_action_loss = (
                                         -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                                         * active_masks_batch
                                 ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        # print(f'Actor Zero grad time: {time.time() - st}')
        st = time.time()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        critic_backward_time = time.time() - st
        # print(f'Actor Backward time: {critic_backward_time}')
        # st = time.time()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()
        # print(f'Actor Step time: {time.time() - st}')
        # st = time.time()

        # critic update
        # print(values.shape, value_preds_batch.shape)
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )

        self.policy.critic_optimizer.zero_grad()
        # print(f'Critic Zero grad time: {time.time() - st}')

        st = time.time()
        critic_loss = (
                value_loss * self.value_loss_coef
        )  # TODO add gradient accumulation here
        critic_loss.backward()
        actor_backward_time = time.time() - st
        # print(f'Critic Backward time: {actor_backward_time}')
        # st = time.time()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()
        # print(f'Critic Step time: {time.time() - st}')
        # print('_'*50)

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
            actor_backward_time,
            critic_backward_time,
        )

    def train(self, buffer: GraphReplayBuffer, update_actor: bool = True):
        """
        Perform a training update using minibatch GD.
        buffer: (GraphReplayBuffer)
            buffer containing training data.
        update_actor: (bool)
            whether to update actor network.

        :return train_info: (dict)
            contains information regarding
            training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0

        for _ in range(self.ppo_epoch):
            st = time.time()
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            actor_backward_time, critic_backward_time = 0, 0

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                    actor_bt,
                    critic_bt,
                ) = self.ppo_update(sample, update_actor)

                actor_backward_time += actor_bt
                critic_backward_time += critic_bt
                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += imp_weights.mean()

            print(f'PPO epoch time: {time.time() - st}')
            print(f'PPO epoch actor backward time: {actor_backward_time}')
            print(f'PPO epoch critic backward time: {critic_backward_time}')

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Convert networks to training mode"""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """Convert networks to eval mode"""
        self.policy.actor.eval()
        self.policy.critic.eval()


def minibatchGenerator(
        obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor, max_batch_size: int
):
    """
    Split a big batch into smaller batches.
    """
    num_minibatches = obs.shape[0] // max_batch_size + 1
    for i in range(num_minibatches):
        yield (
            obs[i * max_batch_size: (i + 1) * max_batch_size],
            node_obs[i * max_batch_size: (i + 1) * max_batch_size],
            adj[i * max_batch_size: (i + 1) * max_batch_size],
            agent_id[i * max_batch_size: (i + 1) * max_batch_size],
        )


class GR_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    args: argparse.Namespace
        Arguments containing relevant model information.
    obs_space: (gym.Space)
        Observation space.
    node_obs_space: (gym.Space)
        Node observation space
    edge_obs_space: (gym.Space)
        Edge dimension in graphs
    action_space: (gym.Space)
        Action space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(self, device, obs_dim, num_services, svc_obs_dim=5, svc_adj=None) -> None:
        super(GR_Actor, self).__init__()
        self.hidden_size = 64

        self._gain = 0.01
        self._use_orthogonal = True
        self._use_policy_active_masks = True
        self._use_naive_recurrent_policy = False
        self._use_recurrent_policy = True
        self._recurrent_N = 1
        self.split_batch = False
        self.max_batch_size = 32
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.svc_adj = svc_adj

        node_obs_shape = obs_dim
        # returns (num_nodes, num_node_feats)
        edge_dim = 1  # returns (edge_dim,)

        gat_dim = 1
        self.gnn_base = GNNBase(node_obs_shape, edge_dim, 'node')
        self.gatv2 = DependencyAtt(svc_obs_dim, gat_dim, 1)
        gnn_out_dim = self.gnn_base.out_dim  # output shape from gnns
        mlp_base_in_dim = 32 + obs_dim
        self.base = MLPBase(obs_shape=obs_dim, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(self.hidden_size, self._use_orthogonal, self._gain, num_services)

        self.to(device)

    def forward(self, obs, node_obs, adj, agent_id, rnn_states, masks, available_actions=None, deterministic=False,
                svc_obs=None, svc_adj=None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        :param svc_adj:
        :param obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        :param node_obs: (np.ndarray / torch.Tensor):
            Local agent graph node features to the actor.
        :param adj: (np.ndarray / torch.Tensor):
            Adjacency matrix for the graph
        :param agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        :param rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        :param deterministic: (bool)
            Whether to sample from action distribution or return the mode.
        :param svc_obs: (np.ndarray / torch.Tensor)
            Observation for mircoservices.

        :return actions: (torch.Tensor)
            Actions to take.
        :return action_log_probs: (torch.Tensor)
            Log probabilities of taken actions.
        :return rnn_states: (torch.Tensor)
            Updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        svc_obs = check(svc_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        svc_adj = check(svc_adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            # nbd_features = torch.cat([node_obs[0], torch.zeros(8, 6)], -1)
            # svc_features = self.gatv2(svc_obs, svc_adj)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        return (actions, action_log_probs, rnn_states)

    def evaluate_actions(self, obs, node_obs, adj, agent_id, rnn_states, action, masks, available_actions=None,
                         active_masks=None, svc_obs=None, svc_adj=None) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability and entropy of given actions.
        :param svc_adj:
        :param obs: (torch.Tensor)
            Observation inputs into network.
        :param node_obs: (torch.Tensor):
            Local agent graph node features to the actor.
        :param adj: (torch.Tensor):
            Adjacency matrix for the graph.
        :param agent_id: (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        :param action: (torch.Tensor)
            Actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor)
            If RNN network, hidden states for RNN.
        :param masks: (torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        :param available_actions: (torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        :param active_masks: (torch.Tensor)
            Denotes whether an agent is active or dead.
        :param svc_obs: (torch.Tensor)

        :return action_log_probs: (torch.Tensor)
            Log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor)
            Action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        svc_obs = check(svc_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        svc_adj = check(svc_adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'eval Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            svc_adj = svc_adj[0].unsqueeze(0).repeat(adj.shape[0], 1, 1)
            svc_features = self.gatv2(svc_obs, svc_adj)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        return (action_log_probs, dist_entropy)


class GR_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions
    given centralized input (MAPPO) or local observations (IPPO).
    args: (argparse.Namespace)
        Arguments containing relevant model information.
    cent_obs_space: (gym.Space)
        (centralized) observation space.
    node_obs_space: (gym.Space)
        node observation space.
    edge_obs_space: (gym.Space)
        edge observation space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(self, device, obs_dim, svc_obs_dim=5, svc_adj=None) -> None:
        super(GR_Critic, self).__init__()
        self.hidden_size = 64
        self._use_orthogonal = True
        self._use_naive_recurrent_policy = False
        self._use_recurrent_policy = True
        self._recurrent_N = 1
        self._use_popart = False
        self.split_batch = False
        self.max_batch_size = 32
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.svc_adj = svc_adj
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        edge_dim = 1  # (edge_dim,)

        # TODO modify output of GNN to be some kind of global aggregation
        self.gnn_base = GNNBase(obs_dim, edge_dim, 'global')
        gnn_out_dim = self.gnn_base.out_dim
        # if node aggregation, then concatenate aggregated node features for all agents
        # otherwise, the aggregation is done for the whole graph
        mlp_base_in_dim = 32

        self.base = MLPBase(1, override_obs_dim=mlp_base_in_dim)
        self.gatv2 = DependencyAtt(svc_obs_dim, self.hidden_size, 2)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, node_obs, adj, agent_id, rnn_states, masks, svc_obs=None,
                svc_adj=None) -> Tuple[Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        :param svc_adj:
        :param cent_obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        :param node_obs: (np.ndarray):
            Local agent graph node features to the actor.
        :param adj: (np.ndarray):
            Adjacency matrix for the graph.
        :param agent_id: (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        :param rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if RNN states
            should be reinitialized to zeros.
        :param svc_obs:

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        svc_obs = check(svc_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        svc_adj = check(self.svc_adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (cent_obs.shape[0] > self.max_batch_size):
            # print(f'Cent obs: {cent_obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                cent_obs, node_obs, adj, agent_id, self.max_batch_size
            )
            critic_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                critic_feats_batch = self.base(act_feats_batch)
                critic_features.append(critic_feats_batch)
            critic_features = torch.cat(critic_features, dim=0)
        else:
            nbd_features = self.gnn_base(
                node_obs, adj, agent_id
            )  # CHECK from where are these agent_ids coming
            svc_adj = svc_adj[0].unsqueeze(0).repeat(adj.shape[0], 1, 1)
            svc_features = self.gatv2(svc_obs, svc_adj)
            critic_features = torch.cat([nbd_features], -1)
            critic_features = self.base(critic_features)  # Cent obs here

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return (values, rnn_states)


class DependencyAtt(nn.Module):
    def __init__(self, num_features, hidden_size=64, heads=3):
        super(DependencyAtt, self).__init__()
        self.gatv2 = GAT(num_features, hidden_size, heads=heads, num_layers=1, dropout=0.6, v2=True)
        self.pooling = TopKPooling(in_channels=hidden_size, ratio=0.6)

    def forward(self, x, adj):
        """
        adj: (batch_size,node,node)
        x: (batch_size,node,features)
        :param x:
        :param adj:
        """
        datalist = []
        for i in range(x.size(0)):
            index = adj[i].nonzero(as_tuple=True)
            edge_attr = adj[index]
            edge_index = torch.stack(index, dim=0)
            datalist.append(Data(x=x[i], edge_index=edge_index, edge_attr=edge_attr))
        loader = DataLoader(datalist, shuffle=False, batch_size=x.size(0))
        data = next(iter(loader))
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        x = self.gatv2(x, edge_index, edge_attr)
        x, edge_index, _, batch, _, _ = self.pooling(x, edge_index, edge_attr, batch)
        x, _ = to_dense_batch(x, batch)
        x_global_pooling = x.mean(1)
        return x_global_pooling


"""GNN modules"""


class EmbedConv(MessagePassing):
    def __init__(
            self,
            input_dim: int,
            num_embeddings: int,
            embedding_size: int,
            hidden_size: int,
            layer_N: int,
            use_orthogonal: bool,
            use_ReLU: bool,
            use_layerNorm: bool,
            add_self_loop: bool,
            edge_dim: int = 0,
    ):
        """
            EmbedConv Layer which takes in node features, node_type (entity type)
            and the  edge features (if they exist)
            `entity_embedding` is concatenated with `node_features` and
            `edge_features` and are passed through linear layers.
            The `message_passing` is similar to GCN layer

        Args:
            input_dim (int):
                The node feature dimension
            num_embeddings (int):
                The number of embedding classes aka the number of entity types
            embedding_size (int):
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the linear layers
            layer_N (int):
                Number of linear layers for aggregation
            use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer
            use_ReLU (bool):
                Whether to use reLU for each layer
            use_layerNorm (bool):
                Whether to use layerNorm for each layer
            add_self_loop (bool):
                Whether to add self loops in the graph
            edge_dim (int, optional):
                Edge feature dimension, If zero then edge features are not
                considered. Defaults to 0.
        """
        super(EmbedConv, self).__init__(aggr="add")
        self._layer_N = 1
        self._add_self_loops = False
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        layer_norm = [nn.Identity(), nn.LayerNorm(hidden_size)][use_layerNorm]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)
        self.lin1 = nn.Sequential(
            init_(nn.Linear(input_dim + edge_dim, hidden_size)),
            active_func,
            layer_norm,
        )
        self.lin_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), active_func, layer_norm
        )

        self.lin2 = get_clones(self.lin_h, self._layer_N)

    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
    ):
        if self._add_self_loops and edge_attr is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: OptTensor):
        """
        The node_obs obtained from the environment
        is actually [node_features, node_num, entity_type]
        x_i' = AGG([x_j, EMB(ent_j), e_ij] : j \in \mathcal{N}(i))
        """
        if edge_attr is not None:
            node_feat = torch.cat([x_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([x_j], dim=1)
        x = self.lin1(node_feat)
        for i in range(self._layer_N):
            x = self.lin2[i](x)
        return x


class TransformerConvNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_embeddings: int,
            embedding_size: int,
            hidden_size: int,
            num_heads: int,
            concat_heads: bool,
            layer_N: int,
            use_ReLU: bool,
            graph_aggr: str,
            global_aggr_type: str,
            embed_hidden_size: int,
            embed_layer_N: int,
            embed_use_orthogonal: bool,
            embed_use_ReLU: bool,
            embed_use_layerNorm: bool,
            embed_add_self_loop: bool,
            max_edge_dist: float,
            edge_dim: int = 1,
    ):
        """
            Module for Transformer Graph Conv Net:
            • This will process the adjacency weight matrix, construct the binary
                adjacency matrix according to `max_edge_dist` parameter, assign
                edge weights as the weights in the adjacency weight matrix.
            • After this, the batch data is converted to a PyTorch Geometric
                compatible dataloader.
            • Then the batch is passed through the graph neural network.
            • The node feature output is then either:
                • Aggregated across the graph to get graph encoded data.
                • Pull node specific `message_passed` hidden feature as output.

        Args:
            input_dim (int):
                Node feature dimension
                NOTE: a reduction of `input_dim` by 1 will be carried out
                internally because `node_obs` = [node_feat, entity_type]
            num_embeddings (int):
                The number of embedding classes aka the number of entity types
            embedding_size (int):
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the attention layers
            num_heads (int):
                Number of heads in the attention layer
            concat_heads (bool):
                Whether to concatenate the heads in the attention layer or
                average them
            layer_N (int):
                Number of attention layers for aggregation
            use_ReLU (bool):
                Whether to use reLU for each layer
            graph_aggr (str):
                Whether we want to pull node specific features from the output or
                perform global_pool on all nodes.
                Choices: ['global', 'node']
            global_aggr_type (str):
                The type of aggregation to perform if `graph_aggr` is `global`
                Choices: ['mean', 'max', 'add']
            embed_hidden_size (int):
                Hidden layer size of the linear layers in `EmbedConv`
            embed_layer_N (int):
                Number of linear layers for aggregation in `EmbedConv`
            embed_use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer in `EmbedConv`
            embed_use_ReLU (bool):
                Whether to use reLU for each layer in `EmbedConv`
            embed_use_layerNorm (bool):
                Whether to use layerNorm for each layer in `EmbedConv`
            embed_add_self_loop (bool):
                Whether to add self loops in the graph in `EmbedConv`
            max_edge_dist (float):
                The maximum edge distance to consider while constructing the graph
            edge_dim (int, optional):
                Edge feature dimension, If zero then edge features are not
                considered in `EmbedConv`. Defaults to 1.
        """
        super(TransformerConvNet, self).__init__()
        self.active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        self.num_heads = 3
        self.concat_heads = False
        self.edge_dim = 1
        self.max_edge_dist = 1
        self.graph_aggr = 'global'
        self.global_aggr_type = 'mean'
        # NOTE: reducing dimension of input by 1 because
        # node_obs = [node_feat, entity_type]
        self.embed_layer = EmbedConv(
            input_dim=input_dim,
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            hidden_size=embed_hidden_size,
            layer_N=embed_layer_N,
            use_orthogonal=embed_use_orthogonal,
            use_ReLU=embed_use_ReLU,
            use_layerNorm=embed_use_layerNorm,
            add_self_loop=embed_add_self_loop,
            edge_dim=edge_dim,
        )
        self.gnn1 = TransformerConv(
            in_channels=embed_hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
            concat=concat_heads,
            beta=False,
            dropout=0.0,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True,
        )
        self.gnn2 = nn.ModuleList()
        # for i in range(layer_N):
        #     self.gnn2.append(
        #         self.addTCLayer(self.getInChannels(16), 16)
        #     )

    def forward(self, node_obs: Tensor, adj: Tensor, agent_id: Tensor):
        """
        node_obs: Tensor shape:(batch_size, num_nodes, node_obs_dim)
            Node features in the graph formed wrt agent_i
        adj: Tensor shape:(batch_size, num_nodes, num_nodes)
            Adjacency Matrix for the graph formed wrt agent_i
            NOTE: Right now the adjacency matrix is the distance
            magnitude between all entities so will have to post-process
            this to obtain the edge_index and edge_attr
        agent_id: Tensor shape:(batch_size) or (batch_size, k)
            Node number for agent_i in the graph. This will be used to
            pull out the aggregated features from that node
        """
        # convert adj to edge_index, edge_attr and then collate them into a batch
        batch_size = node_obs.shape[0]
        datalist = []
        for i in range(batch_size):
            edge_index, edge_attr = self.processAdj(adj[i])
            # if edge_attr is only one dimensional
            if len(edge_attr.shape) == 1:
                edge_attr = edge_attr.unsqueeze(1)
            datalist.append(
                Data(x=node_obs[i], edge_index=edge_index, edge_attr=edge_attr)
            )
        loader = DataLoader(datalist, shuffle=False, batch_size=batch_size)
        data = next(iter(loader))
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        if self.edge_dim is None:
            edge_attr = None

        # forward pass through embedConv
        x = self.embed_layer(x, edge_index, edge_attr)

        x1 = x
        # forward pass through first transfomerConv
        x = self.active_func(self.gnn1(x, edge_index, edge_attr))

        # forward pass conv layers
        for i in range(len(self.gnn2)):
            x = self.active_func(self.gnn2[i](x, edge_index, edge_attr))

        # x is of shape [batch_size*num_nodes, out_channels]
        # convert to [batch_size, num_nodes, out_channels]
        x, mask = to_dense_batch(x, batch)

        # only pull the node-specific features from output
        if self.graph_aggr == "node":
            x = self.gatherNodeFeats(x, agent_id)  # shape [batch_size, out_channels]
        # perform global pool operation on the node features of the graph
        elif self.graph_aggr == "global":
            x = self.graphAggr(x)
        return x

    def addTCLayer(self, in_channels: int, out_channels: int):
        """
        Add TransformerConv Layer

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        Returns:
            TransformerConv: returns a TransformerConv Layer
        """
        return TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=self.num_heads,
            concat=self.concat_heads,
            beta=False,
            dropout=0.0,
            edge_dim=self.edge_dim,
            root_weight=True,
        )

    def getInChannels(self, out_channels: int):
        """
        Given the out_channels of the previous layer return in_channels
        for the next layer. This depends on the number of heads and whether
        we are concatenating the head outputs
        """
        return out_channels + (self.num_heads - 1) * self.concat_heads * (out_channels)

    def processAdj(self, adj: Tensor):
        """
        Process adjacency matrix to filter far away nodes
        and then obtain the edge_index and edge_weight
        `adj` is of shape (batch_size, num_nodes, num_nodes)
            OR (num_nodes, num_nodes)
        """
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)
        # filter far away nodes and connection to itself
        # connect_mask = ((adj < self.max_edge_dist) * (adj > 0)).float()
        # adj = adj * connect_mask

        index = adj.nonzero(as_tuple=True)
        edge_attr = adj[index]

        # sparse_to_dense源码
        if len(index) == 3:
            batch = index[0] * adj.size(-1)
            index = (batch + index[1], batch + index[2])

        return torch.stack(index, dim=0), edge_attr

    def gatherNodeFeats(self, x: Tensor, idx: Tensor):
        """
        The output obtained from the network is of shape
        [batch_size, num_nodes, out_channels]. If we want to
        pull the features according to particular nodes in the
        graph as determined by the `idx`, use this
        Refer below link for more info on `gather()` method for 3D tensors
        https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

        Args:
            x (Tensor): Tensor of shape (batch_size, num_nodes, out_channels)
            idx (Tensor): Tensor of shape (batch_size) or (batch_size, k)
                indicating the indices of nodes to pull from the graph

        Returns:
            Tensor: Tensor of shape (batch_size, out_channels) which just
                contains the features from the node of interest
        """
        out = []
        batch_size, num_nodes, num_feats = x.shape
        idx = idx.long()
        for i in range(idx.shape[1]):
            idx_tmp = idx[:, i].unsqueeze(-1)  # (batch_size, 1)
            assert idx_tmp.shape == (batch_size, 1)
            idx_tmp = idx_tmp.repeat(1, num_feats)  # (batch_size, out_channels)
            idx_tmp = idx_tmp.unsqueeze(1)  # (batch_size, 1, out_channels)
            gathered_node = x.gather(1, idx_tmp).squeeze(
                1
            )  # (batch_size, out_channels)
            out.append(gathered_node)
        out = torch.cat(out, dim=1)  # (batch_size, out_channels*k)
        # out = out.squeeze(1)    # (batch_size, out_channels*k)

        return out

    def graphAggr(self, x: Tensor):
        """
        Aggregate the graph node features by performing global pool


        Args:
            x (Tensor): Tensor of shape [batch_size, num_nodes, num_feats]
            aggr (str): Aggregation method for performing the global pool

        Raises:
            ValueError: If `aggr` is not in ['mean', 'max']

        Returns:
            Tensor: The global aggregated tensor of shape [batch_size, num_feats]
        """
        if self.global_aggr_type == "mean":
            return x.mean(dim=1)
        elif self.global_aggr_type == "max":
            max_feats, idx = x.max(dim=1)
            return max_feats
        elif self.global_aggr_type == "add":
            return x.sum(dim=1)
        else:
            raise ValueError(f"`aggr` should be one of 'mean', 'max', 'add'")


class GNNBase(nn.Module):
    """
    A Wrapper for constructing the Base graph neural network.
    This uses TransformerConv from Pytorch Geometric
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv
    and embedding layers for entity types
    Params:
    args: (argparse.Namespace)
        Should contain the following arguments
        num_embeddings: (int)
            Number of entity types in the env to have different embeddings
            for each entity type
        embedding_size: (int)
            Embedding layer output size for each entity category
        embed_hidden_size: (int)
            Hidden layer dimension after the embedding layer
        embed_layer_N: (int)
            Number of hidden linear layers after the embedding layer")
        embed_use_ReLU: (bool)
            Whether to use ReLU in the linear layers after the embedding layer
        embed_add_self_loop: (bool)
            Whether to add self loops in adjacency matrix
        gnn_hidden_size: (int)
            Hidden layer dimension in the GNN
        gnn_num_heads: (int)
            Number of heads in the transformer conv layer (GNN)
        gnn_concat_heads: (bool)
            Whether to concatenate the head output or average
        gnn_layer_N: (int)
            Number of GNN conv layers
        gnn_use_ReLU: (bool)
            Whether to use ReLU in GNN conv layers
        max_edge_dist: (float)
            Maximum distance above which edges cannot be connected between
            the entities
        graph_feat_type: (str)
            Whether to use 'global' node/edge feats or 'relative'
            choices=['global', 'relative']
    node_obs_shape: (Union[Tuple, List])
        The node observation shape. Example: (18,)
    edge_dim: (int)
        Dimensionality of edge attributes
    """

    def __init__(
            self,
            node_obs_shape: int,
            edge_dim: int,
            graph_aggr: str,
    ):
        super(GNNBase, self).__init__()

        self.hidden_size = 16
        self.heads = 3
        self.concat = False
        self.gnn = TransformerConvNet(
            input_dim=node_obs_shape,
            edge_dim=edge_dim,
            num_embeddings=3,
            embedding_size=2,
            hidden_size=32,
            num_heads=3,
            concat_heads=False,
            layer_N=2,
            use_ReLU=True,
            graph_aggr=graph_aggr,
            global_aggr_type='mean',
            embed_hidden_size=16,
            embed_layer_N=1,
            embed_use_orthogonal=True,
            embed_use_ReLU=True,
            embed_use_layerNorm=True,
            embed_add_self_loop=False,
            max_edge_dist=1,
        )

    def forward(self, node_obs: Tensor, adj: Tensor, agent_id: Tensor):
        x = self.gnn(node_obs, adj, agent_id)
        return x

    @property
    def out_dim(self):
        return self.hidden_size + (self.heads - 1) * self.concat * (self.hidden_size)
