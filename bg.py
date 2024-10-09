import numpy as np
import copy

from util import *

class Backgammon:
    def __init__(self):
        self.idx_count = 26
        self.action_size = 26 ** 4

    def get_initial_board(self):
        return np.array(INIT_BOARD)

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, board, player):
        if player == -1:
            board = np.flip(-board)
        return board

    def get_next_state(self, board, play, player):
        for move in play:
            self.make_move(board, move)
        return board

    def check_win(self, board):
        return np.all(board >= 0) or np.all(board <= 0)

    def get_value_and_terminated(self, board):
        '''Always returns 1 for value'''
        if self.check_win(board):
            return 1, True
        return 0, False

    def get_encoded_state(self, board, jumps):
        # [26] + jumps
        # (7, 26) + (6)

        white_cnt = np.maximum(board, 0).astype(np.float32) / 15
        white_one = (board == 1).astype(np.float32)
        white_tower = (board > 1).astype(np.float32)
        black_cnt = np.maximum(-board, 0).astype(np.float32) / 15
        black_one = (board == -1).astype(np.float32)
        black_tower = (board < -1).astype(np.float32)
        # Using small threshold for float comparison
        empty = (board == 0).astype(np.float32)

        # Stack the arrays vertically
        encoded_board = np.vstack(
            (white_cnt, white_one, white_tower, black_cnt, black_one, black_tower, empty))

        jumps_encoded = np.zeros(4, dtype=np.float32)
        jumps_encoded[:len(jumps)] = jumps
        jumps_encoded = jumps_encoded/6

        indicies = np.arange(26)
        black_pip = np.sum(black_cnt * indicies) / 200
        indicies = 25-indicies
        white_pip = np.sum(white_cnt * indicies) / 200

        features = np.concatenate(
            [jumps_encoded, [white_pip, black_pip]]).astype(np.float32)

        return encoded_board, features

    def get_encoded_states_batched(self, boards, jumps):
        # Encode boards
        white_cnt = np.maximum(boards, 0).astype(np.float32) / 15
        white_one = (boards == 1).astype(np.float32)
        white_tower = (boards > 1).astype(np.float32)
        black_cnt = np.maximum(-boards, 0).astype(np.float32) / 15
        black_one = (boards == -1).astype(np.float32)
        black_tower = (boards < -1).astype(np.float32)
        empty = (boards == 0).astype(np.float32)

        encoded_boards = np.stack(
            [white_cnt, white_one, white_tower, black_cnt, black_one, black_tower, empty], axis=1)

        # Encode features
        jumps_encoded = jumps.astype(np.float32) / 6

        indices = np.arange(26)
        black_pip = np.sum(black_cnt * indices, axis=1) / 200
        white_pip = np.sum(white_cnt * (25 - indices), axis=1) / 200

        encoded_features = np.column_stack(
            [jumps_encoded, white_pip, black_pip]).astype(np.float32)

        return encoded_boards, encoded_features

    def get_valid_plays(self, board, jumps, player):
        if len(jumps) == 4:
            return self._generate_plays_quads(board, jumps)
        return self._generate_plays(board, jumps)

    def plays_to_actions(self, plays):
        actions = np.zeros(self.action_size)
        for play in plays:
            play_idx = []
            for move in sorted(play, key=lambda move: (move[1]-move[0])):
                play_idx.append(move[0])
            while len(play_idx) < 4:
                play_idx.append(25)

            action_idx = playidx_to_action(play_idx)
            actions[action_idx] = 1
        return actions

    def play_to_action(self, play):
        play = sorted(play, key=lambda move: (move[1]-move[0]))
        play_idx = []
        for move in play:
            play_idx.append(move[0])
        while len(play_idx) < 4:
            play_idx.append(25)
        return playidx_to_action(play_idx)
    
    def make_move(self, board, move) -> None:
        '''Apply move on board'''
        start, end = move

        if board[start] == 0:
            raise ValueError(
                f"Invalid move {move}: No piece to move at start position. Board: \n{self.draw(board)}")
        board[start] -= 1
        # Handle bearoff
        if end >= 25:
            return board
        # Handle hitting opponent checker
        if board[end] == -1:
            board[end] = 1
            board[25] -= 1
        elif board[end] < -1:
            raise ValueError(
                f"Invalid move {move}: End position has more than -1 checker. Board: \n{self.draw(board)}")
        # Handle regular move
        else:
            board[end] += 1
        return board

    def is_valid_move(self, board, jumps, move):
        # print(board)
        start, end = move[0], move[1]

        jump = abs(end - start)
        if jump not in jumps:
            return False

        # Check if start exists
        # print(board)
        # print(move)
        if board[start] <= 0:
            return False

        # Check if move is in right direction
        if end <= start:
            return False

        # Check if jailed
        if board[0] > 0:
            if start != 0:
                return False

        # Check if trying to bear off
        if end >= 25:
            if np.all(board[0:19] <= 0):
                if end == 25:
                    return True
                elif np.all(board[19:start] <= 0):
                    return True
                return False
            else:
                return False

        # Check if end is blocked
        if board[end] < -1:
            return False
        return True

    def _generate_plays(self, board, jumps):
        # print("Generating plays")
        board = board
        jumps = jumps

        def get_movable_pieces(board):
            if board[0] > 0:
                return [0]
            moveable_pieces = []
            for i in range(len(board)):
                if board[i] > 0:
                    moveable_pieces.append(i)
            return moveable_pieces

        res = []

        play = []

        def dfs(board, jumps):
            # print("called dfs")
            movable_pieces = get_movable_pieces(board)
            # print(f"movable_pieces: {get_movable_pieces(board)} with jumps {jumps}")
            if movable_pieces == [] or jumps == []:
                res.append(copy.deepcopy(play))
                return

            for piece in movable_pieces:
                for j in range(len(jumps)):
                    move = (piece, piece+jumps[j])  # make negative for black
                    # print(board, movable_pieces, move)
                    # print("hi")
                    if self.is_valid_move(board, jumps, move) == False:
                        # print("Invalid move")
                        res.append(copy.deepcopy(play))
                        continue

                    # print(f"board: {board}, jumps={jumps}")
                    # print(f"move: {move}")
                    tmp_board = board.copy()
                    tmp_board = self.make_move(tmp_board, move)

                    tmp_jump = copy.copy(jumps)
                    tmp_jump.pop(j)
                    # print(f"tmp_board: {tmp_board}, tmp_jump={tmp_jump}")
                    # print()

                    play.append(move)

                    dfs(tmp_board, tmp_jump)
                    play.pop()

        def remove_duplicate_plays(res):
            res = [tuple(play) for play in res]
            res = set(res)
            res = list(res)
            res = [list(play) for play in res]
            return res

        def sort_plays(res):
            for i in range(len(res)):
                res[i] = sorted(res[i], key=lambda x: (x[0], x[1]))
            return res

        dfs(board, jumps)
        # print(f"{len(res)} plays generated by dfs")
        # print(res)
        if not res:
            return []
        res = sort_plays(res)
        res = remove_duplicate_plays(res)
        # print(res)
        # res.sort(key=lambda x: [x[0][0], x[1][0], x[0][1]])
        max_length_play = max(len(play) for play in res)
        res = [play for play in res if len(play) == max_length_play]

        return res

    def _generate_plays_quads(self, board, jumps):

        def get_movable_pieces(board):
            if board[0] > 0:
                return [0]
            moveable_pieces = []
            for i in range(len(board)):
                if board[i] > 0:
                    moveable_pieces.append(i)
            return moveable_pieces

        res = []
        play = []

        def dfs(board, jumps):
            movable_pieces = get_movable_pieces(board)
            if movable_pieces == [] or jumps == []:
                res.append(copy.deepcopy(play))
                return

            for piece in movable_pieces:
                move = (piece, piece+jumps[0])

                if self.is_valid_move(board, jumps, move) == False:
                    res.append(copy.deepcopy(play))
                    continue

                # print(f"board: {board}, jumps={jumps}")
                # print(f"move: {move}")
                tmp_board = board.copy()
                tmp_board = self.make_move(tmp_board, move)
                tmp_jump = copy.copy(jumps)
                tmp_jump.pop()
                # print(f"tmp_board: {tmp_board}, tmp_jump={tmp_jump}")
                # print()
                play.append(move)

                dfs(tmp_board, tmp_jump)
                play.pop()
        dfs(board, jumps)

        def sort_plays(res):
            for i in range(len(res)):
                res[i] = sorted(res[i], key=lambda x: (x[0]))
            return res
        sorted_res = sort_plays(res)

        def remove_duplicate_plays(res):
            res = [tuple(play) for play in res]
            res = set(res)
            res = list(res)
            res = [list(play) for play in res]
            return res
        processed_res = remove_duplicate_plays(sorted_res)

        if not processed_res:
            return []

        max_length_play = max(len(play) for play in processed_res)
        longest_plays = [play for play in processed_res if len(
            play) == max_length_play]

        return longest_plays

    def draw(self, board) -> str:
        '''String representation of the board'''
        def transform(i):
            x = str(i)
            if x[0] != "-":
                x = " " + x
            if len(x) < 3:
                x += " "
            return x

        # top index numbers for the board
        top_idx = "  " + \
            " ".join(f"{transform(i)}" for i in range(
                12, 0, -1)) + "   " + transform(0)

        # bottom index numbers for the board
        bot_idx = "  " + \
            " ".join(f"{transform(i)}" for i in range(13, 25)) + \
            "   " + transform(25)

        # ===== boarder for top and bottom
        boarder = "="*57

        # top row of the board from board
        top = "||" + \
            " ".join(f"{transform(board[i])}" for i in range(12, 6, -1)) + "|" + \
            " ".join(f"{transform(board[i])}" for i in range(6, 0, -1)) + "|| " + \
            str(transform(board[0]))

        # bottom row of the board from board
        bot = "||" + \
            " ".join(f"{transform(board[i])}" for i in range(13, 19)) + "|" + \
            " ".join(f"{transform(board[i])}" for i in range(19, 25)) + "|| " + \
            str(transform(board[25]))

        # x combines all the strings together to make the board
        x = top_idx + "\n" + boarder + "\n" + top + \
            "\n\n\n" + bot + "\n" + boarder + "\n" + bot_idx

        # print(x)
        return x
