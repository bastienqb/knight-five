"""Unit tests for the game."""

from pytest import fixture

from knight_five.game import EXAMPLE_BOARD, Game


@fixture
def game():
    return Game(board=EXAMPLE_BOARD.board, goal=EXAMPLE_BOARD.goal, start=EXAMPLE_BOARD.start)


def test_excel_pos(game):
    assert game.excel_pos(pos=(3, 0)) == "a1"
    assert game.excel_pos(pos=(0, 0)) == "a4"
    assert game.excel_pos(pos=(0, 3)) == "d4"


def test_jump(game):
    assert game._jump(pos=(0, -1), jump=(1, 0)) == (1, -1)


def test_get_all_actions(game):
    # checking with set since do not care about the order
    assert set(game.get_all_actions()) == {
        None,
        (-2, -1, 0),
        (-2, 1, 0),
        (-2, 0, -1),
        (-2, 0, 1),
        (-1, -2, 0),
        (-1, 2, 0),
        (-1, 0, -2),
        (-1, 0, 2),
        (0, -2, -1),
        (0, -2, 1),
        (0, -1, -2),
        (0, -1, 2),
        (0, 1, -2),
        (0, 1, 2),
        (0, 2, -1),
        (0, 2, 1),
        (2, -1, 0),
        (2, 1, 0),
        (2, 0, -1),
        (2, 0, 1),
        (1, -2, 0),
        (1, 2, 0),
        (1, 0, -2),
        (1, 0, 2),
    }


def test_get_sink_rise(game):
    same_alt, pos_diam, rate = game._get_sink_rise(pos=(0, 0))
    assert set(same_alt) == {(0, 0), (0, 2)}
    assert pos_diam == (3, 3)
    assert rate == 1 / 2


def test_get_sink_rise_diam_same_altitude(game):
    game._board[3, 2] = 10.0
    same_alt, pos_diam, rate = game._get_sink_rise(pos=(0, 1))
    assert set(same_alt) == {(0, 1), (2, 0)}
    assert pos_diam is None
    assert rate == 1 / 2
