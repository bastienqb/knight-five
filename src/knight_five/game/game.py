"""Game class."""

import logging
from itertools import chain
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

Jump = Tuple[int, int, int]
Position = Tuple[int, int]

SHAPE = 2
ALLOWED_FREQ = 3
COLS = "abcdefghijklmopqrstu"

logger = logging.getLogger(__name__)


class Game:
    """Represent a game instance."""

    def __init__(self, board: NDArray, goal: Position, start: Position) -> None:
        """Create a new game, which is not initialised yet."""
        assert len(board.shape) == SHAPE, "board should be in 2D"
        assert board.shape[0] == board.shape[1], "board should be squared"

        self._dim_board = board.shape[0]
        logger.debug(f"board dim is {self._dim_board}x{self._dim_board}")

        self._board = board.copy()
        self._goal = goal
        logger.debug(f"king needs to reach {self._goal}")

        self._king_pos = start  # king starts at bottom left of the board
        logger.debug(f"king starts at {self.king_pos}")
        self._freq = np.zeros_like(board)  # to count how many times you come to a tile

        self._actions = self.get_all_actions()
        logger.debug(f"number of actions is {len(self._actions)}")

        self._sink = None
        self._sink_pairs = None
        self._rise = None
        self._rate = None
        self._has_jump = False

        self.is_on = False

    @property
    def king_pos(self) -> Position:
        """Position (x, y) of the king."""
        return self._king_pos

    def excel_pos(self, pos: Position) -> str:
        """Return position with the game convention."""
        return f"{COLS[pos[1]]}{self._dim_board - pos[0]}"

    def initialize(self) -> None:
        """Initialise the game."""
        self.is_on = True
        self._has_jump = True  # convention to trigger the computation if decides to stay
        logger.debug("game is initialized")

    def step(self, action: Union[Jump, None]) -> None:
        """Step the game."""
        logger.debug(f"position is {self.king_pos}")
        if action is not None:
            logger.debug(f"jump is {action}")
            self._king_pos = self._jump(pos=self._king_pos, jump=action)
            self._has_jump = True
            logger.debug(f"new position is {self.king_pos}")
            # if king reaches the goal, stop the game
            if self.king_pos == self._goal:
                self.is_on = False
                logger.debug("game finished")
                return
        else:
            # stay for 1 minute
            logger.debug("no jump")
            self._step_board()
            self._has_jump = False
            logger.debug(f"same position is {self.king_pos}")

    def _jump(self, pos: Position, jump: Jump) -> Position:
        """Update the position after the jump."""
        return (pos[0] + jump[0], pos[1] + jump[1])

    def get_all_actions(self) -> List[Jump]:
        """Create a list of all possible actions."""
        # by convention the first action is to do nothing
        actions = [None]

        jump_set = {0, 1, 2}

        for i in range(-2, 3):
            j_left = jump_set.difference({abs(i)})
            for j in set(chain.from_iterable([[j, -j] for j in j_left])):
                k_left = j_left.difference({abs(j)})
                for k in set(chain.from_iterable([[k, -k] for k in k_left])):
                    actions.append((i, j, k))
        return actions

    def _get_sink_rise(self, pos: Position) -> Tuple[List[Position], Position, float]:
        """Compute sink and rise lattices, and the rate."""
        pos_diam = self._get_diam_opposite(pos)
        if pos_diam == pos:
            # not possible with even dimension
            pos_diam = None
        same_alt = set(map(tuple, np.argwhere(self._board == self._board[pos])))

        if (pos_diam is not None) and (pos_diam in same_alt):
            same_alt.discard(pos_diam)
            pos_diam = None

        return same_alt, pos_diam, 1 / len(same_alt)

    def _set_sink_rise(self, pos: Position) -> None:
        """Set sink and rice lattices, and the rate if king stays."""
        same_alt, self._rise, self._rate = self._get_sink_rise(pos=pos)

        self._sink_pairs = same_alt
        self._sink = (
            np.array([p[0] for p in same_alt]),
            np.array([p[1] for p in same_alt]),
        )

    def _get_diam_opposite(self, pos: Position) -> Position:
        return (self._dim_board - 1 - pos[0], self._dim_board - 1 - pos[1])

    def get_valid_actions(self, pos: Position) -> [bool]:
        """Return the index of all valid actions."""
        valid_actions = []
        if self._has_jump:
            # if it has just jumped, compute the potential sink/rise if it stays
            self._set_sink_rise(pos=pos)
        for i, action in enumerate(self._actions):
            # for "stay" action, check if it is a blocking position
            if action is None:
                if self._can_stay(pos=pos):
                    valid_actions.append(i)
                continue
            if self._is_valid_jump(jump=action, pos=pos):
                valid_actions.append(i)
        return valid_actions

    def _is_valid_jump(self, jump: Jump, pos: Position) -> bool:
        return (
            # stay within board boundaries
            self._stay_in_board(jump=jump, pos=pos)
            # altitude is matching
            and self._correct_altitude(jump=jump, pos=pos)
            # cannot visit a lattice more than 3 times
            and self._respect_freq(jump=jump, pos=pos)
        )

    def _can_stay(self, pos: Position) -> bool:
        """Indicate if the position is a dead end (one step roll-out for stay action)."""
        for jump in self._actions[1:]:  # by convention stay action is the first
            if (not self._stay_in_board(jump=jump, pos=pos)) or (not self._respect_freq(jump=jump, pos=pos)):
                continue
            new_pos = self._jump(pos=pos, jump=jump)
            alt_new = self._board[new_pos]
            alt_cur = self._board[pos]
            if new_pos in self._sink_pairs:
                # same altitude and reachable
                return True
            if new_pos == self._rise:
                # if not too high and same multiple
                if (alt_new <= alt_cur + 2) and (
                    ((alt_cur - alt_new + 2) / 2 % self._rate)
                    or ((alt_cur - alt_new + 1) / 2 % self._rate)
                    or ((alt_cur - alt_new) / 2 % self._rate)
                    or ((alt_cur - alt_new - 1) / 2 % self._rate)
                    or ((alt_cur - alt_new - 2) / 2 % self._rate)
                ):
                    return True
                continue
            if (alt_new <= alt_cur + 2) and (
                ((alt_cur - alt_new + 2) % self._rate)
                or ((alt_cur - alt_new + 1) % self._rate)
                or ((alt_cur - alt_new) % self._rate)
                or ((alt_cur - alt_new - 1) % self._rate)
                or ((alt_cur - alt_new - 2) % self._rate)
            ):
                return True
        # if no action returns True, then blocked
        return False

    def _stay_in_board(self, jump: Jump, pos: Position) -> bool:
        new_pos = self._jump(pos=pos, jump=jump)
        return all((i > -1) and (i < self._dim_board) for i in new_pos)

    def _correct_altitude(self, jump: Jump, pos: Position) -> bool:
        new_pos = self._jump(pos=pos, jump=jump)
        return self._board[pos] + jump[2] == self._board[new_pos]

    def _respect_freq(self, jump: Jump, pos: Position) -> bool:
        new_pos = self._jump(pos=pos, jump=jump)
        return self._freq[new_pos] < ALLOWED_FREQ

    def _step_board(self) -> None:
        if self._has_jump:
            self._freq[self.king_pos] += 1
        self._board[self._sink] -= self._rate
        if self._rise is not None:
            self._board[self._rise] += self._rate
