"""A PPO agent."""

from typing import Tuple

import numpy as np
import torch
from gymnasium.core import ObsType
from torch import nn
from torch.distributions.categorical import Categorical

TF_BOARD = "events"
TORCH_PARAMS = "torch_params"
SAVE_WEIGHTS = "weights"
SAVE_OPTIMISER = "optimiser-state"
SAVE_UPDATES = "update-{weight_update_idx}"


class MaskedCategorical(Categorical):
    """Categorical distribution with masked actions."""

    def __init__(self, logits: torch.Tensor | None = None, action_mask: torch.Tensor | None = None):
        """Compute logist by masking actions."""
        self._action_mask = action_mask
        if action_mask is not None:
            mask_value = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype)
            logits = torch.where(self._action_mask, logits, mask_value)
        super().__init__(logits=logits)

    def entropy(self):
        """Compute the entropy of the distribution."""
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        p_log_p_masked = torch.where(self._action_mask, p_log_p, torch.tensor(0, dtype=p_log_p.dtype))

        return -p_log_p_masked.sum(-1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialise the network weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNetwork(nn.Module):
    """Standard PPO nn."""

    def __init__(self, conv_dim: tuple[int, int, int], fc_dim: tuple[int], action_dim: int) -> None:
        """Initialise the netwoek architecture."""
        super().__init__()

        self.conv_feat = nn.Sequential(
            layer_init(nn.Conv2d(conv_dim[0], 6, 3, padding="same")),
            nn.ReLU(),
            layer_init(nn.Conv2d(6, 12, 2, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(12 * 2 * 2, 120)),
            nn.ReLU(),
        )

        self.fc_feat = nn.Sequential(layer_init(nn.Linear(fc_dim[0], 32)), nn.ReLU())

        self.actor = layer_init(nn.Linear(152, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(152, 1), std=1)

    def _forward_obs(self, conv_obs: torch.Tensor, fc_obs: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.conv_feat(conv_obs), self.fc_feat(fc_obs)), dim=1)

    def get_value(
        self,
        conv_obs: torch.Tensor,
        fc_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the value of the state, using the critic network."""
        return self.critic(self._forward_obs(conv_obs=conv_obs, fc_obs=fc_obs))

    def get_action_entropy_value(
        self,
        conv_obs: torch.Tensor,
        fc_obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the action, its log-probability, the entropy of the distribution and the value of the state."""
        feat = self._forward_obs(conv_obs=conv_obs, fc_obs=fc_obs)
        logits = self.actor(feat)
        probs = MaskedCategorical(logits=logits, action_mask=action_mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(feat)


def obs_to_torch(obs: ObsType, eval: bool = False, device: str | None = None) -> dict[str, torch.Tensor]:
    """Transfer obs to torch tensors."""
    dic = {
        "conv": torch.tensor(obs["conv"], dtype=torch.float32).to(device),
        "fc": torch.tensor(obs["fc"], dtype=torch.float32).to(device),
        "action_mask": torch.tensor(obs["action_mask"], dtype=torch.bool).to(device),
    }
    if eval:  # meaning only one obs
        dic = {obs_name: obs.unsqueeze(0) for obs_name, obs in dic.items()}
    return dic
