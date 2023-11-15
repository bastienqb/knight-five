"""Game class."""

from itertools import chain
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

Jump = Tuple[int, int, int]
Position = Tuple[int, int]

SHAPE = 2
ALLOWED_FREQ = 3


class Game:
    """Represent a game instance."""

    def __init__(self, board: NDArray, goal: Position, start: Position) -> None:
        """Create a new game, which is not initialised yet."""
        assert len(board.shape) == SHAPE, "board should be in 2D"
        assert board.shape[0] == board.shape[1], "board should be squared"

        self._dim_board = board.shape[0]

        self._board = board.copy()
        self._goal = goal

        self._king_pos = start  # king starts at bottom left of the board
        self._freq = np.zeros_like(board)  # to count how many times you come to a tile

        self._actions = self.get_all_actions()

        self._sink = None
        self._rise = None
        self._rate = None
        self._has_jump = False

        self.is_on = False

    @property
    def king_pos(self) -> Position:
        """Position (x, y) of the king."""
        return self._king_pos

    def initialize(self) -> None:
        """Initialise the game."""
        self.is_on = True
        self._has_jump = True  # convention to trigger the computation if decides to stay

    def step(self, action: Union[Jump, None]) -> None:
        """Step the game."""
        if action is not None:
            self.jump(jump=action)
            self._has_jump = True
            # if king reaches the goal, stop the game
            if self.king_pos == self._goal:
                self.is_on = False
                return
        else:
            # stay for 1 minute
            self._step_board()
            self._has_jump = False

    def jump(self, jump: Jump) -> None:
        """Update the position after the jump."""
        if not self._is_valid_action(jump=jump, pos=self.king_pos):
            raise ValueError(f"the jump {jump} is not valid")
        self._king_pos = (self._king_pos[0] + jump[0], self._king_pos[1] + jump[1])

    def get_all_actions(self) -> List[Jump]:
        """Create a list of all possible actions."""
        actions = []

        jump_set = {0, 1, 2}

        for i in range(-2, 3):
            j_left = jump_set.difference({abs(i)})
            for j in set(chain.from_iterable([[j, -j] for j in j_left])):
                k_left = j_left.difference({abs(j)})
                for k in set(chain.from_iterable([[k, -k] for k in k_left])):
                    actions.append((i, j, k))
        return actions

    def _set_sink_rise(self, pos: Position) -> None:
        pos_diam = self._get_diam_opposite(pos)
        if pos_diam == pos:
            # not possible with even dimension
            pos_diam = None
        same_alt = set(map(tuple, np.argwhere(self._board == self._board[pos])))

        if (pos_diam is not None) and (pos_diam in same_alt):
            same_alt.discard(pos_diam)
            pos_diam = None

        self._sink = (
            np.array([p[0] for p in same_alt]),
            np.array([p[1] for p in same_alt]),
        )
        self._rate = 1 / len(same_alt)  # never divide by 0 since there is always at least one element
        self._rise = pos_diam

    def _get_diam_opposite(self, pos: Position) -> Position:
        return (self._dim_board - 1 - pos[0], self._dim_board - 1 - pos[1])

    def get_valid_actions(self, pos: Position) -> [bool]:
        """Return the index of all valid actions."""
        valid_actions = []
        for i, action in enumerate(self._actions):
            if self._is_valid_action(jump=action, pos=pos):
                valid_actions.append(i)
        return valid_actions

    def _is_valid_action(self, jump: Jump, pos: Position) -> bool:
        new_pos = (pos[0] + jump[0], pos[1] + jump[1])
        return (
            # stay within board boundaries
            all((i > -1) and (i < self._dim_board) for i in new_pos)
            # altitude is matching
            and (self._board[pos] + jump[2] == self._board[new_pos])
            # cannot visit a lattice more than 3 times
            and (self._freq[new_pos] < ALLOWED_FREQ)
        )

    def _step_board(self) -> None:
        if self._has_jump:
            self._set_sink_rise(pos=self.king_pos)
            self._freq[self.king_pos] += 1
        self._board[self._sink] -= self._rate
        if self._rise is not None:
            self._board[self._rise] += self._rate
