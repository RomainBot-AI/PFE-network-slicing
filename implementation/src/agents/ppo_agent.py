#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
 MODULE : src/agents/ppo_agent.py
 OBJET  : Agent d Apprentissage par Renforcement PPO Multi-Binaire PyTorch
====================================================================================================

DESCRIPTION DÉTAILLÉE :
-----------------------
Implémente l'agent PPO (Proximal Policy Optimization) avec architecture Acteur-Critique en PyTorch.
  - Acteur Multi-Binaire : Produit une distribution de Bernoulli pour chaque tranche spécialisée (p_i in [0, 1]).
  - Critique : Estime la fonction de valeur V(s) pour le calcul de l'avantage (GAE - Generalized Advantage Estimation).
  - PPO Loss : Clipped Surrogate Objective (epsilon = 0.2) + MSE Loss pour le Critique.

====================================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import numpy as np
from typing import Tuple, List, Dict, Any


class ActorCritic(nn.Module):
    """
    Réseau Neuronal Acteur-Critique PyTorch pour Actions Multi-Binaires.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Tête Acteur (Politique pi_theta)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Sortie p_i in [0, 1] pour chaque tranche
        )

        # Tête Critique (Fonction de valeur V_phi)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value.squeeze(-1)


class PPOAgent:
    """
    Agent PPO pour l'optimisation dynamique des tranches réseau 5G/6G.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        batch_size: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Échantillonne un vecteur d'actions binaires c_init in {0, 1}^K selon la distribution de Bernoulli.
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs, value = self.policy(state_t)
            dist = Bernoulli(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        action_np = action.squeeze(0).cpu().numpy().astype(np.int32)
        return action_np, float(log_prob.item()), float(value.item())

    def update(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        log_probs: List[float],
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ):
        """
        Mise à jour des poids PPO via Clipped Loss et GAE.
        """
        if len(states) == 0:
            return

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32)
        old_log_probs_t = torch.tensor(np.array(log_probs), dtype=torch.float32)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32)
        values_t = torch.tensor(np.array(values), dtype=torch.float32)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32)

        # Calcul des avantages GAE (Generalized Advantage Estimation)
        advantages = torch.zeros_like(rewards_t)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            next_val = values_t[t + 1] if t + 1 < len(rewards) else 0.0
            next_non_terminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + self.gamma * next_val * next_non_terminal - values_t[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns_t = advantages + values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimisation PPO
        for _ in range(self.ppo_epochs):
            probs, state_values = self.policy(states_t)
            dist = Bernoulli(probs)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)

            ratios = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns_t)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
