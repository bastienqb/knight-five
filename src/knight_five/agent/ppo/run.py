"""Visualise a trainded PPO agent."""


import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium
import rich_click as click
import torch
from rich import print
from rich.progress import track

from ...gym.game import KnightGame, KnightObsWrapper
from ...gym.game_example import get_env
from .agent import SAVE_UPDATES, SAVE_WEIGHTS, TORCH_PARAMS, ActorCriticNetwork, obs_to_torch

RENDER = False
WEIGHTS_UPDATE = None
WEIGHTS_DIR = None
LOG = False
NUM_RUNS = 1
RUN_FOLDER = "trials"

obs_to_torch_ = partial(obs_to_torch, eval=True, device="cpu")


def run_game(env: gymnasium.Env, agent: ActorCriticNetwork) -> dict[str, Any]:
    """Run a game."""
    next_obs, info = env.reset()
    next_obs = obs_to_torch_(next_obs)
    cur_pos = info["knight"]
    cur_min = info["minutes"]
    moves = []

    while True:
        action, _, _, _ = agent.get_action_entropy_value(
            conv_obs=next_obs["conv"],
            fc_obs=next_obs["fc"],
            action_mask=next_obs["action_mask"],
        )
        next_obs, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
        next_obs = obs_to_torch_(next_obs)
        if info["knight"] != cur_pos:
            moves.append((info["minutes"] - cur_min, cur_pos))
            cur_pos = info["knight"]
            cur_min = info["minutes"]
        if terminated or truncated:
            break
    return {
        "minutes": info["minutes"],
        "moves": moves,
        "success": info["success"],
    }


@click.command(context_settings={"show_default": True})
@click.option("-e", "--env-name", type=str, help="environment to use")
@click.option("-w", "--weights-dir", type=str, default=WEIGHTS_DIR, help="path to the set of weights to use")
@click.option(
    "--weights-update",
    type=str,
    default=WEIGHTS_UPDATE,
    help="weight update to use. By default, use the most recent",
)
@click.option("-n", "--num-runs", type=int, default=NUM_RUNS, help="nomber of games to run")
@click.option("-r", "--render", is_flag=True, default=RENDER, help="toogle whether or not to visualise the run")
@click.option("--log-run", is_flag=True, default=LOG, help="toggle to log the trajectory in txt file")
@click.pass_context
def run(
    ctx: click.Context,
    env_name: str,
    weights_dir: str | None = WEIGHTS_DIR,
    weights_update: int | None = WEIGHTS_UPDATE,
    num_runs: int = NUM_RUNS,
    render: bool = RENDER,
    log_run: bool = LOG,
) -> None:
    """Run the PPO agent."""
    base_env = get_env(env_name=env_name)
    render_mode = None
    if render:
        render_mode = "human"

    path_xp = Path(weights_dir).resolve(strict=True)
    if weights_update is None:
        weights_path = max((path_xp / TORCH_PARAMS).iterdir(), key=lambda s: int(s.name.split("-")[1]))
        weights_update = weights_path.name.split("-")[1]
        print(f":warning: no update given. Auto-selected last update {weights_update}")
    else:
        weights_path = path_xp / TORCH_PARAMS / SAVE_UPDATES.format(weight_update_idx=weights_update)
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

    res = []
    for i in track(range(num_runs), description="Games are running..."):
        run_res = run_game(env=env, agent=agent)
        print(f"🀄 run {i} {'success' if run_res['success'] else 'fail'}: {run_res['minutes']} min")
        res.append(run_res)

    env.close()

    if log_run:
        trials_folder = path_xp / RUN_FOLDER
        trials_folder.mkdir(exist_ok=True, parents=True)
        file_path = trials_folder / f"trial__{weights_update}__{time.time()}.json"
        with file_path.open("w") as f:
            json.dump(res, f)
        print(f"💾 results saved at {file_path}")
