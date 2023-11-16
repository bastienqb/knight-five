from knight_five.gym.game import KnightGame
from knight_five.gym.game_example import EXAMPLE_BOARD

if __name__ == "__main__":
    env = KnightGame(board=EXAMPLE_BOARD.board, goal=EXAMPLE_BOARD.goal, start=EXAMPLE_BOARD.start, render_mode="human")

    obs, info = env.reset()

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

    for s in steps:
        action = all_actions.index(s)
        obs, reward, terminated, truncated, info = env.step(action)

    print(env.minutes)

    env.close()
