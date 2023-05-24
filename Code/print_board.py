import numpy as np
BOARD_SIZE = 6
MAX_TURN = 100
N_ROWS = 1200
COLORS = {'R': '0', 'B': '1', 'G': '2', 'Y': '3', 'P': '4'}
ACTION_SPACE = [((i, j), (i, j + 1)) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE - 1)] + \
               [((i + 1, j), (i, j)) for i in range(BOARD_SIZE - 1)
                for j in range(BOARD_SIZE)]  # 生成所有可能的动作


def check(arr):
        repeats = set()
        for i in range(0, BOARD_SIZE - 2):
            for j in range(BOARD_SIZE):
                if arr[i, j] != 'nan' and (arr[i+1:i+3, j] == arr[i, j]).all():
                    repeats.add((i+1, j))
                if arr[j, i] != 'nan' and (arr[j, i+1:i+3] == arr[j, i]).all():
                    repeats.add((j, i+1))
        return repeats

def check2(arr):
        repeats = set()
        for i in range(BOARD_SIZE):
            row = arr[i]
            row_view = np.lib.stride_tricks.as_strided(row, (BOARD_SIZE - 2, 3), (8, 8))
            connected = np.all(row_view[:, 1:] == row_view[:, :-1], axis=1)
            equals = np.where(connected)[0]
            repeats.update([(i, j + 1) for j in equals])
            
            col = arr[:, i]
            col_view = np.lib.stride_tricks.as_strided(col, (BOARD_SIZE - 2, 3), (8, 8))
            connected = np.all(col_view[:, 1:] == col_view[:, :-1], axis=1)
            equals = np.where(connected)[0]
            repeats.update([(j + 1, i) for j in equals])
        return repeats