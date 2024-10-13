import torch
import numpy as np
from util import timed
import time

from nodeParallel import NodeParallel

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, spGames):
        # Get batch of spg boards and jumps
        boards, padded_jumps = [], []
        for spg in spGames:
            boards.append(spg.root.board)
            if len(spg.root.jumps) < 4:
                padded_jumps.append(spg.root.jumps + [0,0])
            else:
                padded_jumps.append(spg.root.jumps)
        boards = np.array(boards, dtype=np.float32)
        padded_jumps = np.array(padded_jumps, dtype=np.float32)

        # Get batch of encoded states
        enc_boards, enc_features = self.game.get_encoded_states_batched(boards, padded_jumps)

        # Get batch of policy and value predictions
        start = time.time()
        policy, value = self.model(
            torch.tensor(enc_boards, device=self.model.device),
            torch.tensor(enc_features, device=self.model.device)
        )
        end = time.time()
        print(f"search(): Forward call time: {end - start:.6f} seconds for {len(spGames)} games")

        policy = torch.softmax(policy, axis=1).detach().cpu().numpy()
        value = value.detach().cpu().numpy()

        # Add dirichlet noise to policy
        noise = np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * noise

        # Expand root nodes
        count = 0
        start = time.time()
        for i, spg in enumerate(spGames):
            if spg.root is None:
                raise ValueError(f"search(): spg.root is None, spg_board: {spg_board}, spg_jumps: {spg_jumps}")
            if spg.root.is_fully_expanded():
                # print(f"search(): spg.root is fully expanded, skipping expand()")
                continue
            spg_board, spg_jumps = spg.root.board, spg.root.jumps
            spg_policy, spg_value = policy[i], float(value[i][0])
            valid_plays = self.game.get_valid_plays(spg_board, spg_jumps, 1)
            valid_actions = self.game.plays_to_actions(valid_plays)
            spg_policy *= valid_actions
            spg_policy /= np.sum(spg_policy)

            spg.root.state_value = spg_value
            spg.root.expand(spg_policy, valid_plays)
            count += 1
        end = time.time()
        # print(f"search(): Expand time: {end - start:.6f} seconds for {count} games")

        for search in range(self.args['num_searches']):
            # Selection
            count = 0
            start = time.time()
            for g in range(len(spGames)):
                spg = spGames[g]
                idx = spg.idx
                node = spg.root
                spg.node = None
                while (node.visit_count[idx] > 0 or node.jumps == []) and node.is_fully_expanded():
                    node = node.select(idx)
                #print(f"search(): Selected node is level {node.level}")
                value, is_terminal = self.game.get_value_and_terminated(node.board)
                value = self.game.get_opponent_value(value)
                if is_terminal:
                    node.backpropagate(value * node.search_weight, idx)
                elif node.state_value is not None:
                    #print(f"Serach(): Found cached node with state_value {node.state_value}, skipping forward + expand.")
                    node.backpropagate(node.state_value * node.search_weight, idx)
                else:
                    #print(f"Search(): calling forward + expand for node with state_value {node.state_value}")
                    count += 1
                    spg.node = node
            end = time.time()
            print(f"search(): Selection time: {end - start:.6f} seconds for {len(spGames)} games. Founds cache for {len(spGames) - count} nodes, need to expand {count} nodes")

            # Parallel Model Call on Expandable SPGs
            start = time.time()
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
            if len(expandable_spGames) > 0:
                boards = []
                jumps = []
                for mappingIdx in expandable_spGames:
                    boards.append(spGames[mappingIdx].node.board)
                    if len(spGames[mappingIdx].node.jumps) < 4:
                        jumps.append(spGames[mappingIdx].node.jumps + [0,0])
                    else:
                        jumps.append(spGames[mappingIdx].node.jumps)
                boards = np.array(boards, dtype=np.float32)
                jumps = np.array(jumps, dtype=np.float32)

                enc_boards, enc_features = self.game.get_encoded_states_batched(boards, jumps)
                policy, value = self.model(
                    torch.tensor(enc_boards, device=self.model.device),
                    torch.tensor(enc_features, device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).detach().cpu().numpy()
                value = value.detach().cpu().numpy()
            end = time.time()
            # if len(expandable_spGames) > 0:
            #     print(f"Search(): Model call time: {end - start:.6f} seconds for {len(expandable_spGames)} games")

            # Expansion and Backpropagation
            start = time.time()
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                idx = spGames[mappingIdx].idx
                spg_policy, spg_value = policy[i], float(value[i][0])
                node.state_value = spg_value

                valid_plays = self.game.get_valid_plays(node.board, node.jumps, 1)
                valid_actions = self.game.plays_to_actions(valid_plays)
                spg_policy *= valid_actions
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy, valid_plays)
                node.backpropagate(spg_value, idx)
            end = time.time()
            # if len(expandable_spGames) > 0:
            #     print(f"Search(): Expansion + Backprop time: {end - start:.6f} seconds for {len(expandable_spGames)} games")
