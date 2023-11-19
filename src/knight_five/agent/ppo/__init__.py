"""Solving the game with PPO."""

import rich_click as click

from .run import run
from .train import train


@click.group()
@click.pass_context
def ppo(ctx):
    """Solve the game with the PPO algorithm."""
    pass


ppo.add_command(train)
ppo.add_command(run)
