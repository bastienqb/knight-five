"""Readme example."""

import rich_click as click
from rich.console import Console

from ..gym.game import KnightGame
from ..gym.game_example import EXAMPLE_BOARD

console = Console()


@click.command()
def readme():
    """Run the Readme example."""
    env = KnightGame(board=EXAMPLE_BOARD.board, goal=EXAMPLE_BOARD.goal, start=EXAMPLE_BOARD.start, render_mode="human")

    _, _ = env.reset()

    all_actions = env.get_all_actions()

    steps = [
        None,  # a0
        (-2, 1, 0),  # b3
        (0, -1, 2),  # a3
        (1, 0, 2),  # a2
        (-2, 1, 0),  # b4
        None,
        None,
        None,
        None,
        None,
        (2, -1, 0),  # a2
        None,
        None,
        None,
        (1, 2, 0),  # c1
        None,
        None,
        (-1, -2, 0),  # a2
        (1, 2, 0),  # c1
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (0, 1, 2),  # d1
        None,
        (-1, 0, 2),  # d2
        (0, -1, 2),  # c2
        None,
        (0, -1, 2),  # b2
        (1, 0, 2),  # b1
        (-2, 0, 1),  # b3
        (0, 1, 2),  # c3
        (-1, 0, 2),  # c4
        (0, -2, 1),  # a4
        (0, 1, 2),  # b4
        (0, 2, 1),  # d4
    ]

    with console.status("Game is running..."):
        for s in steps:
            action = all_actions.index(s)
            _ = env.step(action)

    console.print(f"✨ Game ended! Knight tour took {env.minutes} min.", style="purple")

    env.close()
