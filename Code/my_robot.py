import numpy as np
import time
import random

BOARD_SIZE = 6
MAX_TURN = 100
N_ROWS = 1200
COLORS = {'R': 0, 'B': 1, 'G': 2, 'Y': 3, 'P': 4}
ACTION_SPACE = [((i, j), (i, j + 1)) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE - 1)] + \
               [((i + 1, j), (i, j)) for i in range(BOARD_SIZE - 1)
                for j in range(BOARD_SIZE)]  # 生成所有可能的动作


class MyBoard:
    """
    这是用于记录棋盘状态的类，包含棋盘的大小、棋盘的状态、棋盘的颜色。包含三个方法：
    1. change：交换棋盘上两个位置的棋子
    2. check：检查棋盘上是否有三个或以上的连续棋子
    3. eliminate：消除棋盘上的连续棋子（TODO 可以尝试能否用numpy向量化加速）
    正常来说不需要改动里面的代码。
    """

    def __init__(self, board, colors, turn_number):
        self.size = board.shape[0]
        self.board = board.copy()
        self.colors = colors
        self.turn_number = turn_number

    def copy(self):
        newboard = self.board.copy()
        copied = MyBoard(newboard, self.colors, self.turn_number)
        return copied

    def change(self, loc1, loc2):
        x1, y1 = loc1
        x2, y2 = loc2
        self.board[x1, y1], self.board[x2, y2] = self.board[x2, y2], self.board[x1, y1]

    @staticmethod
    def check(arr):
        repeats = set()
        visited = {(i, j): 0 for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)}
        for i in range(0, BOARD_SIZE - 2):
            for j in range(BOARD_SIZE):
                if visited[(i, j)] == 1 or visited[(j, i)] == 1:
                    continue
                if arr[i, j] != 'nan' and (arr[i+1:i+3, j] == arr[i, j]).all():
                    repeats.add((i+1, j))
                    visited[(i, j)] = 1
                    visited[(i+1, j)] = 1
                    visited[(i+2, j)] = 1
                if arr[j, i] != 'nan' and (arr[j, i+1:i+3] == arr[j, i]).all():
                    repeats.add((j, i+1))
                    visited[(j, i)] = 1
                    visited[(j, i+1)] = 1
                    visited[(j, i+2)] = 1
        return repeats
    
    @staticmethod
    def check2(arr):
        repeats = set()
        for i in range(BOARD_SIZE):
            row = arr[i]
            row_view = np.lib.stride_tricks.as_strided(row, (BOARD_SIZE - 2, 3), (row.itemsize, row.itemsize))
            connected = np.all(row_view == row_view[:, 0, None], axis=1)
            equals = np.where(connected)[0]
            repeats.update([(i, j + 1) for j in equals])
            
            col = arr[:, i]
            col_view = np.lib.stride_tricks.as_strided(col, (BOARD_SIZE - 2, 3), (col.itemsize*6, col.itemsize*6))
            connected = np.all(col_view == col_view[:, 0, None], axis=1)
            equals = np.where(connected)[0]
            repeats.update([(j + 1, i) for j in equals])
        return repeats

    def game_over(self):
        if self.turn_number >= MAX_TURN:
            return True
        elif (self.board[:self.size, :self.size] == 'nan').any():
            return True
        return False

    def eliminate(self, func=lambda x: (x - 2) ** 2):
        arr = self.board[:BOARD_SIZE, :BOARD_SIZE]
        to_eliminate = np.zeros((self.size, self.size), dtype=int)
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        to_visit = self.check(arr)
        score = 0
        
        for coord in to_visit:
            if to_eliminate[coord[0], coord[1]] == 1:
                continue
            head = 0
            connected = [coord, ]
            while head < len(connected):
                current = connected[head]
                to_eliminate[current[0], current[1]] = 1
                for d in directions:
                    neighbor = current + d
                    if (neighbor < 0).any() or (neighbor >= self.size).any():
                        continue
                    if (arr[neighbor[0], neighbor[1]] == arr[current[0], current[1]]
                            and to_eliminate[neighbor[0], neighbor[1]] == 0):
                        connected.append(neighbor)
                head += 1
            score += func(len(connected))

        col_eliminated = np.sum(to_eliminate, axis=1)
        col_remained = self.size - col_eliminated
        for i in range(self.size):
            if col_eliminated[i] == 0:
                continue
            col = self.board[i]
            self.board[i, :col_remained[i]] = col[:self.size][to_eliminate[i] == 0]
            self.board[i, col_remained[i]:N_ROWS - col_eliminated[i]] = col[self.size:]

        # Return the total score and the number of columns eliminated
        return score, col_eliminated


class Agent:
    """
    这是智能体类，包含选择函数。
    智能体的参数包括当前的棋盘状态，当前下一步可以引发消除的动作，当前双方的得分和回合数。
    目前我的设计思路：
    （0）虽然我们有全部的棋盘信息，但我感觉其实不需要一次读太长，可能读目前的棋盘+2-3次搜索需要的棋盘（感觉在30个左右）就可以了。
    （1）计概A学过的MinMax + AB剪枝算法，其中局面的估值（可能可以）采用这一状态下的得分减去对手的得分（假如游戏胜利则来一个大的），这样可以保证智能体的行动是为了自己的利益最大化。
    """

    def __init__(self, board, operations, scores, turn_number):
        self.board = board
        self.operations = operations
        self.scores = scores
        self.turn_number = turn_number

    def minimax(self, board, depth, max_player, action_space, scores, alpha=-np.inf, beta=np.inf):
        '''
        区域中间和周围的差异，棋盘连通块的数目
        '''
        if depth == 0 or board.game_over():
            return scores[0] - scores[1], action_space[0]
        
        best_action = action_space[0]
        if max_player:
            best_value = -9999
            t1 = time.time()
            for i in range(60):
                t2 = time.time()
                if t2 - t1 > 0.45:
                    break
                new_board = board.copy()
                action = action_space[i]
                new_board.change(*action)
                if i == 0:
                    t3 = time.time()
                total_score, columns_eliminated = new_board.eliminate()
                while columns_eliminated.sum() and not (new_board.board[:BOARD_SIZE, :BOARD_SIZE] == 'nan').any():
                    score, columns_eliminated = new_board.eliminate()
                    total_score += score
                if i == 0:
                    t4 = time.time()
                    print("Elimination time 2: ", t4 - t3)
                new_scores = scores.copy()
                new_scores[0] += total_score  # update max player's score
                value, _ = self.minimax(
                    new_board, depth - 1, False, ACTION_SPACE, new_scores, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
            print("Best value: ", best_value)
        else:
            best_value = 9999
            for i in range(60):
                new_board = board.copy()
                action = action_space[i]
                new_board.change(*action)
                total_score, columns_eliminated = new_board.eliminate()
                while columns_eliminated.sum() and not (new_board.board[:BOARD_SIZE, :BOARD_SIZE] == 'nan').any():
                    score, columns_eliminated = new_board.eliminate()
                    total_score += score
                new_scores = scores.copy()
                new_scores[1] += total_score  # update min player's score
                value, _ = self.minimax(
                    new_board, depth - 1, True, ACTION_SPACE, new_scores, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
                if alpha >= beta:
                    break
        return best_value, best_action

    def select(self, depth=2):
        """
        TODO 写一个选择函数，目前有3种选择（MinMax + AB剪枝、MCTS（蒙特卡洛树搜索）、Q-Learning（这个我来试着写一写）
        输入：当前棋盘、当前可行的操作
        """
        op = self.operations
        no_op = [x for x in ACTION_SPACE if x not in op]
        op = op + no_op
        return self.minimax(self.board, depth, True, op, self.scores)[1]



class Plaser:
    """
    游戏玩家类，包含移动函数。
    """

    def __init__(self, is_First):
        self.is_First = is_First

    def move(self, board, operations, scores, turn_number):
        """
        这是移动函数，输入是棋盘、操作、分数、回合数，输出是移动的位置。
        """
        my_board = MyBoard(board=board, colors=np.array(
            list(COLORS.keys())), turn_number=turn_number)
        root = Agent(board=my_board, operations=operations,
                     scores=scores, turn_number=turn_number)
        t2 = time.time()
        action = root.select()
        t3 = time.time()
        print("selection time: ", t3 - t2)
        return action
