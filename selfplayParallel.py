import numpy as np
import time
import os
import logging
import gc
import psutil 

from mctsParallel import MCTSParallel
from nodeParallel import NodeParallel
from util import timed, roll_dice

class SPG:
    def __init__(self,root, idx):
        self.root = root
        self.idx = idx
        self.memory = []
        self.node = None
       
@timed
def selfPlayParallel(model, game, args, board=None, jumps=None):
    logger = logging.getLogger('backgammon_ai')
    return_memory = []
    player = 1
    board = board if board is not None else game.get_initial_board()
    jumps = jumps if jumps is not None else roll_dice()
    mctsParallel = MCTSParallel(game, args, model)
    rootParallel = NodeParallel(game, args, board, jumps, visit_count=1)
    spGames = [SPG(root=rootParallel, idx=idx) for idx in range(args['num_parallel_games'])]
    rootParallel = None
    P1_wins = 0
    P2_wins = 0
    spIter = 1

    while len(spGames) > 0:
        start_time = time.time()
        logger.info(f"selfPlayParallel(): Running for {len(spGames)} games")
        mctsParallel.search(spGames)

        for g in range(len(spGames))[::-1]:
            spg = spGames[g]
            idx = spg.idx

            action_probs = np.zeros(game.action_size)
            for child in spg.root.children:
                action_probs[child.action_taken] = child.visit_count[idx]

            if np.sum(action_probs) == 0:
                print(spg.root)
                for child in spg.root.children:
                    print(child)
                raise ValueError(f"selfPlayParallel(): Sum of spGames action probs is 0, action_probs{action_probs}")

            action_probs /= np.sum(action_probs)
            spg.memory.append((spg.root.board, spg.root.jumps, action_probs, player))

            temperature_action_probs = action_probs  ** (1 / args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            
            # Add this check
            if np.isnan(temperature_action_probs).any():
                print(spg.root)
                for child in spg.root.children:
                    print(child)
                raise ValueError(f"selfPlayParallel(): NaN detected in temperature_action_probs: {temperature_action_probs}, action_probs{action_probs}")

            action = np.random.choice(game.action_size, p=temperature_action_probs)
            # print(f"spg {idx} root before action: {spg.root}")
            # Take action
            for child in spg.root.children:
                if child.action_taken == action:
                    spg.root = child.find_random_child_weighted()
                    spg.root.parent = None
                    break
            # print(f"spg {idx} root after action: {spg.root}")
            if spg.root is None:
                raise ValueError(f"selfPlayParallel(): Chosen child node with action {action} not found")

            spg.board = spg.root.board
            spg.jumps = spg.root.jumps

            value, is_terminal = game.get_value_and_terminated(spg.board)
            if is_terminal:
                if player == 1:
                    P1_wins += 1
                else:
                    P2_wins += 1
                print(f"    {os.getpid()}: selfPlayParallel(): SPgame {idx} Terminal State. Add  {len(spg.memory)} rows of data to return_memory.")
                for hist_neutral_board, hist_jumps, hist_action_probs, hist_player in spg.memory:
                    hist_outcome = value if hist_player == player else game.get_opponent_value(value)
                    #encoded_board, encoded_features = self.game.get_encoded_state(hist_neutral_board, hist_jumps)
                    return_memory.append((
                        hist_neutral_board,
                        hist_jumps,
                        hist_action_probs,
                        hist_outcome
                    ))
                # print(f"SPGame {idx} ended on player {-1*player} with board {spg.root}")
                # print(f"    {os.getpid()}: selfPlayParallel(): SPgame {idx} ended with {len(spg.memory)} states:")
                # print(f"memory[0]: {spg.memory[0][0]}, Player= {spg.memory[0][3]}, V= {return_memory[0][3]}")
                # print(f"memory[1]: {spg.memory[1][0]}, Player= {spg.memory[1][3]}, V= {return_memory[1][3]}")
                # print(f"memory[2]: {spg.memory[2][0]}, Player= {spg.memory[2][3]}, V= {return_memory[2][3]}")
                # print(f"memory[-3]: {spg.memory[-3][0]}, jumps={spg.memory[-3][1]}, Player= {spg.memory[-3][3]}, V= {return_memory[-3][3]}")
                # print(f"memory[-2]: {spg.memory[-2][0]}, jumps={spg.memory[-2][1]}, Player= {spg.memory[-2][3]}, V= {return_memory[-2][3]}")
                # print(f"memory[-1]: {spg.memory[-1][0]}, jumps={spg.memory[-1][1]}, Player= {spg.memory[-1][3]}, V= {return_memory[-1][3]}")
                # print(f"")
                # print(f"deleting spg {spGames[g].idx}: {spGames[g].root}")
                del spGames[g]
        player = game.get_opponent(player)

        end_time = time.time()
        iteration_time = end_time - start_time
        print(f"    {os.getpid()}: selfPlayParallel(): SP iter {spIter} done. time: {iteration_time:.4f} seconds for {len(spGames)} games with {args['num_searches']} num_searches")

        start_gc = time.time()
        collected = gc.collect()
        end_gc = time.time()
        print(f"    {os.getpid()}:GC collected {collected} in time: {end_gc - start_gc:.4f} seconds")
        
        
        spIter += 1
    print(f"{os.getpid()}: selfPlayParallel(): All SPgames ended. P1 wins {P1_wins} P-1 wins {P2_wins}. States: {len(return_memory)}")
    return return_memory