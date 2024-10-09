import torch
import numpy as np

from node import Node


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, board, jumps):
        root = Node(self.game, self.args, board, jumps, visit_count=1)
        device = self.model.device

        enc_board, enc_features = self.game.get_encoded_state(
            root.board, root.jumps)
        policy, value = self.model(torch.tensor(enc_board, device=device).unsqueeze(
            0), torch.tensor(enc_features, device=device).unsqueeze(0))

        value = float(value.item())

        policy = torch.softmax(policy, axis=1).squeeze(
            0).detach().cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_plays = self.game.get_valid_plays(root.board, root.jumps, 1)
        valid_actions = self.game.plays_to_actions(valid_plays)
        policy *= valid_actions
        policy /= np.sum(policy)

        root.state_value = value
        root.expand(policy, valid_plays)

        for search in range(self.args['num_searches']):
            # print(search)
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.board)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                enc_board, enc_features = self.game.get_encoded_state(
                    node.board, node.jumps)
                policy, value = self.model(torch.tensor(enc_board, device=device).unsqueeze(
                    0), torch.tensor(enc_features, device=device).unsqueeze(0))
                value = float(value.item())
                node.state_value = value

                policy = torch.softmax(policy, axis=1).squeeze(
                    0).detach().cpu().numpy()

                valid_plays = self.game.get_valid_plays(
                    node.board, node.jumps, 1)
                valid_actions = self.game.plays_to_actions(valid_plays)
                policy *= valid_actions
                policy /= np.sum(policy)

                node.expand(policy, valid_plays)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs, root
