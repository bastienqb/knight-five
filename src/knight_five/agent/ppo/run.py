"""Visualise a trainded PPO agent."""


from functools import partial
from pathlib import Path

import rich_click as click
import torch
from rich.console import Console

from ...gym.game import KnightGame, KnightObsWrapper
from ...gym.game_example import get_env
from .agent import SAVE_UPDATES, SAVE_WEIGHTS, TORCH_PARAMS, ActorCriticNetwork, obs_to_torch

RENDER = True
WEIGHTS_UPDATE = None


@click.command(context_settings={"show_default": True})
@click.option("-e", "--env-name", type=str, help="environment to use")
@click.option("-w", "--weights-dir", type=str, help="path to the set of weights to use")
@click.option(
    "--weights-update",
    type=str,
    default=WEIGHTS_UPDATE,
    help="weight update to use. By default, use the most recent",
)
@click.option("-r", "--render", is_flag=True, default=RENDER, help="toogle whether or not to visualise the run")
@click.pass_context
def run(
    ctx: click.Context,
    env_name: str,
    weights_dir: str,
    weights_update: int | None = None,
    render: str = RENDER,
) -> None:
    """Run the PPO agent."""
    base_env = get_env(env_name=env_name)
    render_mode = None
    console = Console()
    if render:
        render_mode = "human"

    path_xp = Path(weights_dir).resolve(strict=True) / TORCH_PARAMS
    if weights_update is None:
        weights_path = max(path_xp.iterdir(), key=lambda s: int(s.name.split("-")[1]))
        console.print(f":warning: no update given. Auto-selected last update {weights_path.name.split('-')[1]}")
        weights_path = weights_path / SAVE_WEIGHTS
    else:
        weights_path = path_xp / SAVE_UPDATES.format(weight_update_idx=weights_update)
        weights_path.exists()
        weights_path = weights_path / SAVE_WEIGHTS

    env = KnightObsWrapper(
        KnightGame(board=base_env.board, goal=base_env.goal, start=base_env.start, render_mode=render_mode)
    )
    device = "cpu"
    # custom method to get shape
    obs_shapes = env.get_obs_shape()
    action_shape = env.action_space.n

    agent = ActorCriticNetwork(conv_dim=obs_shapes["conv"], fc_dim=obs_shapes["fc"], action_dim=action_shape).to(device)
    agent.load_state_dict(torch.load(weights_path))
    agent.eval()
    obs_to_torch_ = partial(obs_to_torch, eval=True, device="cpu")

    next_obs, _ = env.reset()
    next_obs = obs_to_torch_(next_obs)

    with console.status("Game is running..."):
        while True:
            action, _, _, _ = agent.get_action_entropy_value(
                conv_obs=next_obs["conv"],
                fc_obs=next_obs["fc"],
                action_mask=next_obs["action_mask"],
            )
            next_obs, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
            next_obs = obs_to_torch_(next_obs)
            if terminated or truncated:
                break

    env.close()
