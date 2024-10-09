import np
import time
import random

np.set_printoptions(linewidth=200)

INIT_BOARD = np.array([0, 2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5,
                       -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0])

UNIQUE_JUMPS = [[1, 1, 1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
                [2, 2, 2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
                [3, 3, 3, 3], [3, 4], [3, 5], [3, 6],
                [4, 4, 4, 4], [4, 5], [4, 6],
                [5, 5, 5, 5], [5, 6],
                [6, 6, 6, 6]]


def timed(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        print(f"{function.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper


def action_to_playidx(action):
    ''' Convert action to playidx'''
    playidx = []
    for i in range(4):
        playidx.append(action % 26)
        action = action // 26
    return playidx


def playidx_to_action(playidx):
    '''Convert playidx to action'''
    action = 0
    for i in range(4):
        action += playidx[i] * (26**i)
    return action


def roll_dice():
    '''returns list of jumps sorted smallest to largest'''
    die1 = random.randint(1, 6)
    die2 = random.randint(1, 6)

    if die1 == die2:
        return [die1] * 4
    else:
        return sorted([die1, die2])
