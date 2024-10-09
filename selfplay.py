import numpy as np

from mcts import MCTS
from node import Node
from util import *


@timed
def selfPlayMC(model, game, args, board=None, jumps=None):
    memory = []
    player = 1
    board = board if board is not None else game.get_initial_board()
    jumps = jumps if jumps is not None else roll_dice()
    root = Node(game, args, board, jumps, visit_count=1)
    mcts = MCTS(game, args, model)
    while True:
        # neutral_board = self.game.change_perspective(board, player)
        action_probs, root = mcts.search(board, jumps)
        memory.append((board, jumps, action_probs, player))

        temperature_action_probs = action_probs ** (1 / args['temperature'])
        temperature_action_probs /= np.sum(temperature_action_probs)
        action = np.random.choice(game.action_size, p=temperature_action_probs)

        child = None
        for child in root.children:
            if child.action_taken == action:
                child = child
                break
        root = child.find_random_child_weighted()
        if root is None:
            raise ValueError(f"Child node with action {action} not found")
        board = root.board
        jumps = root.jumps

        value, is_terminal = game.get_value_and_terminated(board)
        if is_terminal:
            returnMemory = []
            for hist_neutral_board, hist_jumps, hist_action_probs, hist_player in memory:
                hist_outcome = value if hist_player == player else game.get_opponent_value(value)
                encoded_board, encoded_features = game.get_encoded_state(hist_neutral_board, hist_jumps)
                returnMemory.append((
                    encoded_board,
                    encoded_features,
                    hist_action_probs,
                    hist_outcome
                ))

            return returnMemory
        player = game.get_opponent(player)
