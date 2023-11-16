"""Board examples."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoardGame:
    """Board game settings."""

    board: NDArray
    goal: Tuple[int, int]
    start: NDArray


EXAMPLE_BOARD = BoardGame(
    board=np.array(
        [
            [11, 10, 11, 14],
            [8, 6, 9, 9],
            [10, 4, 3, 1],
            [7, 6, 5, 0],
        ],
        dtype=np.float64,
    ),
    goal=(0, 3),
    start=(3, 0),
)


COMPETITIVE_BOARD = BoardGame(
    board=np.array(
        [
            [9, 8, 10, 12, 11, 8, 10, 17],
            [7, 9, 11, 9, 10, 12, 14, 12],
            [4, 7, 5, 8, 8, 6, 13, 10],
            [4, 10, 7, 9, 6, 8, 7, 9],
            [2, 6, 4, 2, 5, 9, 8, 11],
            [0, 3, 1, 4, 2, 7, 10, 7],
            [1, 2, 0, 1, 2, 5, 7, 6],
            [0, 2, 4, 3, 5, 6, 2, 4],
        ],
        dtype=np.float64,
    ),
    goal=(0, 7),
    start=(7, 0),
)
