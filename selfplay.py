import numpy as np

from mcts import MCTS
from node import Node
from util import timed, roll_dice

@timed
def selfPlay(model, game, args, board=None, jumps=None):
    memory = []
    player = 1
    board = board if board is not None else game.get_initial_board()
    jumps = jumps if jumps is not None else roll_dice()
    mcts = MCTS(game, args, model)
    root = Node(game, args, board, jumps, visit_count=1)
    while True:
        # Search and Save Results
        action_probs = mcts.search(root)
        memory.append((board, jumps, action_probs, player))
        
        # Add Noise To Action Probabilities
        temperature_action_probs = action_probs ** (1 / args['temperature'])
        temperature_action_probs /= np.sum(temperature_action_probs)
        action = np.random.choice(game.action_size, p=temperature_action_probs)

        # Take Action
        for child in root.children:
            if child.action_taken == action:
                root = child.find_random_child_weighted()
                root.parent = None
                break
        if root is None:
            raise ValueError(f"Child node with action {action} not found")
        
        # Handle Terminal State
        value, is_terminal = game.get_value_and_terminated(root.board)
        if is_terminal:
            return_memory = []
            for hist_neutral_board, hist_jumps, hist_action_probs, hist_player in memory:
                hist_outcome = value if hist_player == player else game.get_opponent_value(value)
                #encoded_board, encoded_features = game.get_encoded_state(hist_neutral_board, hist_jumps)
                return_memory.append((
                    hist_neutral_board,
                    hist_jumps,
                    hist_action_probs,
                    hist_outcome
                ))
            return return_memory
        
        # Change Player
        player = game.get_opponent(player)
