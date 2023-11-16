"""Game class."""

import logging
import math
from itertools import chain
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from numpy.typing import NDArray

Jump = Tuple[int, int, int]
Position = Tuple[int, int]

Observation = Dict
Info = Dict

EPS = 1e-12

SHAPE = 2
ALLOWED_FREQ = 3
BIG_FLOAT = 1e6
COLS = "abcdefghijklmopqrstu"
MAX_MINUTES = 300

REWARD_JUMP = 0
REWARD_STAY = 1
REWARD_REACH = 10
REWARD_BLOCKED = -10
REWARD_OVERTIME = -10

logger = logging.getLogger(__name__)


class KnightGame(gym.Env):
    """Represent a game instance."""

    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(self, board: NDArray, goal: Position, start: Position, render_mode: str | None = None) -> None:
        """Create a new game, which is not initialised yet."""
        assert len(board.shape) == SHAPE, "board should be in 2D"
        assert board.shape[0] == board.shape[1], "board should be squared"

        self._dim_board = board.shape[0]

        self._board_start = board.copy()
        self._start = start
        self._goal = goal

        self._board = None
        self._knight_pos = None
        self._freq = None

        # Game dynamics
        self._sink = None
        self._sink_pairs = None
        self._rise = None
        self._rate = None
        self._has_jump = False

        self.minutes = None

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None

        # RL part
        self._actions = self.get_all_actions()
        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=-BIG_FLOAT, high=BIG_FLOAT, shape=(self._dim_board, self._dim_board), dtype=float
                ),
                "frequency": spaces.Box(low=0, high=ALLOWED_FREQ, shape=(self._dim_board, self._dim_board), dtype=int),
                "knight": spaces.Box(low=0, high=self._dim_board - 1, shape=(2,), dtype=int),
                "goal": spaces.Box(low=0, high=self._dim_board - 1, shape=(2,), dtype=int),
                "action_mask": spaces.MultiBinary(n=len(self._actions)),
                "time_passed": spaces.Box(low=0, high=1.0, shape=(1,), dtype=float),
            }
        )

    def reset(self, seed: int | None = None) -> Tuple[Observation, Info]:
        """Reset the environment."""
        super().reset(seed=seed)

        self._board = self._board_start.copy()
        self._knight_pos = self._start  # king starts at bottom left of the board
        self._freq = np.zeros_like(self._board)

        self.minutes = 0

        self._has_jump = True  # convention to trigger the computation if decides to stay

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action: int | None = None) -> Tuple[Observation, float, bool, bool, Info]:
        """Step the game and returns the information needed for the RL algorithm."""
        terminated = False
        truncated = False
        reward = REWARD_JUMP

        jump = self._actions[action]
        if jump is not None:
            assert self._is_valid_jump(jump=jump, pos=self._knight_pos)
            self._knight_pos = self._jump(pos=self._knight_pos, jump=jump)
            self._has_jump = True
            # if king reaches the goal, stop the game
            if self._knight_pos == self._goal:
                terminated = True
                reward = REWARD_REACH  # reaching the goal is rewarded
        else:
            # stay for 1 minute
            self._step_board()
            self._has_jump = False
            reward = REWARD_STAY  # staying is rewarded
            self.minutes += 1  # time only pass when you stay still

        obs = self._get_obs()
        if all(~obs["action_mask"]):
            # if the agent is blocked on the new lattice, game over
            terminated = True
            reward = REWARD_BLOCKED

        if self.minutes > MAX_MINUTES:
            terminated = True
            reward = REWARD_OVERTIME

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> Observation:
        """Return the observation for the RL agent."""
        action_mask = self._get_action_mask(self._knight_pos)

        return {
            "board": self._board,
            "frequency": self._freq,
            "knight": self._knight_pos,
            "goal": self._goal,
            "action_mask": action_mask,
            "time_passed": self.minutes / MAX_MINUTES,
        }

    def _get_info(self) -> Info:
        """Return the information for the RL agent."""
        return {}

    def excel_pos(self, pos: Position) -> str:
        """Return position with the game convention."""
        return f"{COLS[pos[1]]}{self._dim_board - pos[0]}"

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

    def _get_action_mask(self, pos: Position) -> NDArray:
        """Return the index of all valid actions."""
        valid_actions = np.zeros((len(self._actions),), dtype=bool)
        if self._has_jump:
            # if it has just jumped, compute the potential sink/rise if it stays
            self._set_sink_rise(pos=pos)
        for i, action in enumerate(self._actions):
            # for "stay" action, check if it is a blocking position
            if action is None:
                res = self._can_stay(pos=pos)
            else:
                res = self._is_valid_jump(jump=action, pos=pos)
            valid_actions[i] = res
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
        return math.isclose(self._board[pos] + jump[2], self._board[new_pos], abs_tol=EPS)

    def _respect_freq(self, jump: Jump, pos: Position) -> bool:
        new_pos = self._jump(pos=pos, jump=jump)
        return self._freq[new_pos] < ALLOWED_FREQ

    def _step_board(self) -> None:
        if self._has_jump:
            self._freq[self._knight_pos] += 1
        self._board[self._sink] -= self._rate
        if self._rise is not None:
            self._board[self._rise] += self._rate

    def render(self):
        """Render the environment."""
        # rendering is only done for human in reset and step
        pass

    def _render_frame(self) -> None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self._dim_board  # The size of a single grid square in pixels

        font = pygame.font.SysFont(None, int(pix_square_size / 3))

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * (np.array((self._goal[1], self._goal[0]))),
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array((self._knight_pos[1], self._knight_pos[0])) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self._dim_board + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            for i in range(self._dim_board):
                for j in range(self._dim_board):
                    text = font.render(f"{self._board.T[i, j]:.2f}", True, (0, 0, 0))
                    text_rect = text.get_rect()
                    text_rect.center = pix_square_size * (np.array((i, j)) + 0.5)
                    self.window.blit(text, text_rect)
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        """Close any open resource needed by the game."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
