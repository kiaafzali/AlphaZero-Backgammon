import torch
import numpy as np
from util import timed

from node import Node

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
            boards.append(spg.board)
            if len(spg.jumps) < 4:
                padded_jumps.append(spg.jumps + [0,0])
            else:
                padded_jumps.append(spg.jumps)
        boards = np.array(boards, dtype=np.float32)
        padded_jumps = np.array(padded_jumps, dtype=np.float32)

        # Get batch of encoded states
        enc_boards, enc_features = self.game.get_encoded_states_batched(boards, padded_jumps)

        # Get batch of policy and value predictions
        policy, value = self.model(
            torch.tensor(enc_boards, device=self.model.device),
            torch.tensor(enc_features, device=self.model.device)
        )

        policy = torch.softmax(policy, axis=1).detach().cpu().numpy()
        value = value.detach().cpu().numpy()

        # Add dirichlet noise to policy
        noise = np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * noise

        # Expand root nodes
        for i, spg in enumerate(spGames):
            if spg.root.is_fully_expanded():
                continue
            spg_board, spg_jumps = spg.board, spg.jumps
            spg_policy, spg_value = policy[i], float(value[i][0])
            valid_plays = self.game.get_valid_plays(spg_board, spg_jumps, 1)
            valid_actions = self.game.plays_to_actions(valid_plays)
            spg_policy *= valid_actions
            spg_policy /= np.sum(spg_policy)

            if spg.root is None:
                spg.root = Node(self.game, self.args, spg_board, spg_jumps, visit_count=1)
            spg.root.state_value = spg_value
            spg.root.expand(spg_policy, valid_plays)

        for search in range(self.args['num_searches']):
            # Selection
            for spg in spGames:
                spg.node = None
                node = spg.root
                while node.is_fully_expanded():
                    node = node.select()
                value, is_terminal = self.game.get_value_and_terminated(node.board)
                value = self.game.get_opponent_value(value)
                if is_terminal:
                    node.backpropagate(value * node.search_weight)
                else:
                    spg.node = node

            # Parallel Model Call on Expandable SPGs
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
            
            # Expansion and Backpropagation
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], float(value[i][0])
                node.state_value = spg_value

                valid_plays = self.game.get_valid_plays(node.board, node.jumps, 1)
                valid_actions = self.game.plays_to_actions(valid_plays)
                spg_policy *= valid_actions
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy, valid_plays)
                node.backpropagate(spg_value)