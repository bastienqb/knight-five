"""CLI interface."""

from pathlib import Path

import rich_click as click

from .agent.example_agent import readme
from .agent.ppo import ppo


@click.group()
@click.pass_context
def cli(ctx):
    """CLI to solve the Knight Five game."""
    ctx.obj = {"cwd": Path.cwd()}


@click.group()
@click.pass_context
def examples(ctx):
    """Run an example of the game."""


examples.add_command(readme)
cli.add_command(examples)
cli.add_command(ppo)


if __name__ == "__main":
    cli()
