import math
import random
import numpy as np

from util import UNIQUE_JUMPS, timed
class Node:
    def __init__(self, game, args, board, jumps, parent=None, action_taken=None, prior=0, visit_count=0, level=0):
        self.game = game
        self.args = args
        self.board = board
        self.jumps = jumps
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.visit_count = visit_count
        self.level = level

        self.children: list[Node] = []
        self.value_sum = 0
        self.state_value = None
        self.search_weight = 1
        if self.jumps and len(self.jumps) == 2:
            self.search_weight = 2

    def __str__(self):
        indent = " "*2*self.level
        board_repr = self.game.draw(board=self.board)
        board_repr = '\n'.join(
            [indent + line for line in board_repr.splitlines()])
        return \
            f"""
{indent}{"-"*60}
{indent}Level: {self.level}, N: {self.visit_count}, val: {self.value_sum:.3f}, prior: {self.prior:.3f}, uct:{self.parent.get_ucb(self) if self.parent else None}, weight: {self.search_weight}, num_children: {len(self.children)}
{board_repr}
{indent}jumps={self.jumps}, state_value={self.state_value}, action_taken={self.action_taken}
{indent}board = {self.board}
{indent}{"-"*60}
"""

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        if self.jumps == []:
            return self.find_random_child_weighted()
        else:
            best_child = None
            best_ucb = -np.inf
            for child in self.children:
                ucb = self.get_ucb(child)
                if ucb > best_ucb:
                    best_child = child
                    best_ucb = ucb

            return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = float("inf")
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    #@timed
    def expand(self, policy, plays):
        # plays = self.game.get_valid_plays(self.board, self.jumps, 1)
        if len(self.children) > 0:
            # print(f"expand():skip already expanded node")
            return
        for play in plays:
            child_action_taken = self.game.play_to_action(play)
            child_board = self.board.copy()
            child_board = self.game.get_next_state(child_board, play, 1)
            child_board = self.game.change_perspective(child_board, player=-1)
            child_prior = float(policy[child_action_taken])

            child = Node(self.game, self.args, board=child_board, jumps=[], parent=self,
                         action_taken=child_action_taken, prior=child_prior, visit_count=0, level=self.level+1)
            self.children.append(child)

            for possible_jump in UNIQUE_JUMPS:
                child_child = Node(self.game, self.args, board=child_board, jumps=possible_jump, parent=child,
                                   action_taken=child_action_taken, prior=1, visit_count=0, level=self.level+2)
                child.children.append(child_child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.jumps == []:
            value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

    def find_random_child_weighted(self):
        if not self.children:
            raise RuntimeError(
                f"find random child weighted called on leaf node {self.node}")
        weights = [child.search_weight for child in self.children]
        selected_child = random.choices(self.children, weights=weights, k=1)[0]
        return selected_child
