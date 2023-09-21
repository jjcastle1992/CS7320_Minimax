# MiniMax - Get score for board

import math
import numpy as np
import time
import copy
COUNT = 0    # use the COUNT variable to track number of boards explored


def show_board(board):
    # displays rows of board
    strings = ["" for i in range(board.shape[0])]
    idx = 0
    for row in board:
        for cell in row:
            if cell == 1:
                s = 'X'
            elif cell == -1:
                s = 'O'
            else:
                s = '_'

            strings[idx] += s
        idx += 1

    # display final board
    for s in strings:
        print(s)

def get_board_one_line(board):
    # returns one line rep of a board
    import math
    npb_flat = board.ravel()
    stop = int(math.sqrt(len(npb_flat)))

    bstr = ''
    for idx in range(len(npb_flat)):
        bstr += (str(npb_flat[idx]) + ' ')
        if (idx + 1) % (stop) == 0:
            bstr += '|'
    return bstr

def evaluate(board):
    """
    Using numpy functions to add values in rows and cols
    If we get a sum equal to size of row,col,diag (plus or minus)
     we have a winner
    :param board: a 2d numpy array (should be a square N x N matrix)
    :return: int representing the game win state:
        1 for X win,
        -1 for O win,
        0 for tie OR game in progress
    """
    win = 0
    row_dim, col_dim = board.shape

    # check for row win
    row_sums = np.sum(board, axis=1)
    for rows in row_sums:
        if abs(rows) == row_dim:
            if (rows < 0):
                win = -1
            else:
                win = 1

    # check for column win
    col_sums = np.sum(board, axis=0)
    for cols in col_sums:
        if abs(cols) == col_dim:
            if (cols < 0):
                win = -1
            else:
                win = 1

    # check for diag win
    diag_sums = []
    down_diag = np.diag(board, k=0)
    up_diag = np.diag(np.fliplr(board))
    diag_sums.append(np.sum(down_diag))
    diag_sums.append(np.sum(up_diag))

    if (row_dim == col_dim):
        for entries in diag_sums:
            if (abs(entries) == row_dim):
                if (entries < 1):
                    win = -1
                else:
                    win = 1

    # print(f'\nOur Board:\n{board}')
    # print(f'Row Sums: {row_sums}')
    # print(f'Col Sums: {col_sums}')
    # print(f'Down Diag {down_diag}')
    # print(f'Up Diag {up_diag}')

    return win

def is_terminal_node(board):
    """Evaluates board to determine if a win or terminal state has been
    reached.
    :param board: a 2d numpy array (should be a square N x N matrix)
    :return: Bool where True means either:
        1. A Win state has been achieved (X/O win)
        2. A tie has been reached (due to a terminal board config being
        reached)
    """
    global boards_explored
    boards_explored += 1
    terminal_board = False
    row_dim, col_dim = np.shape(board)
    max_moves = row_dim * col_dim

    # Call Evaluate to determine if a win state has been reached
    game_state = evaluate(board)

    if ((game_state == -1) or (game_state == 1)):
        terminal_board = True

    # if not, then check number of remaining moves (0s on board)
    moves_made = np.count_nonzero(board)
    if(moves_made >= max_moves):
        terminal_board = True

    if(terminal_board):
        print(f'FINAL BOARD CHOSEN:')
        print(board)

    return terminal_board


def get_child_boards(board, char):
    """Gets all children for a possible board.
    :param board: a 2d numpy array (should be a square N x N matrix)
    :param char: a char representing X or O to define which player the
    board should be generated for.
    :return: a list 2d numpy arrays of all the possible child boards
    """

    if not char in ['X', 'O']:
        raise ValueError("get_child_boards: expecting char='X' or 'O' ")

    x_or_o = -1
    if char == 'X':
        x_or_o = 1

    child_list = []

    # find all the spots where 0s exist (empty spots), and fill with
    # char
    print(f'Starting Board:\n{board}')
    print(f'Possible Child Boards:\n')
    for row_idx, row in enumerate(board):
        for column_idx, element in enumerate(row):
            if element == 0:
                possible_board = copy.deepcopy(board)
                possible_board[row_idx][column_idx] = x_or_o
                print(f'{possible_board}\n')
                child_list.append(possible_board)

    return child_list


def minimax(board, depth, maximizingPlayer):
    '''returns the value of the board
       0 (draw) 1 (win for X) -1 (win for O)
       Explores all child boards for this position and returns
       the best score given that all players play optimally
    '''
    if depth == 0 or is_terminal_node(board):
        return evaluate(board)

    if maximizingPlayer:  # max player plays X
        maxEva = -math.inf
        print('For X Turn')
        child_list = get_child_boards(board, 'X')
        for child_board in child_list:
            eva = minimax(child_board, depth-1, False)
            maxEva = max(maxEva, eva)
        return maxEva

    else:             # minimizing player
        minEva = math.inf
        print('For O Turn')
        child_list = get_child_boards(board, 'O')
        for child_board in child_list:
            eva = minimax(child_board, depth - 1, True)
            minEva = min(minEva, eva)
        return minEva

def run_minimax(board_name, board):
    """
    Function designed to call minimax
    :param board_name: a string for the start board name to be tested
    :param board: a 2d numpy array (should be a square N x N matrix)
    """

    print(f'\nRunning Test for Board {board_name}')
    print(f"--------\nStart Board: \n{board}")

    # set max_depth  to the number of blanks (zeros) in the board
    max_depth = np.count_nonzero(board == 0)  # counts moves left (# 0s)
    print(f"Running minimax w/ max depth {max_depth} for:")
    show_board(board)

    if(np.sum(board) > 0):
        is_x_to_move = False
    else:
        is_x_to_move = True

    # read time before and after call to minimax for b1
    tic = time.perf_counter()
    score = minimax(board, max_depth, is_x_to_move)
    toc = time.perf_counter()
    print(f'TESTING Board {board_name}')
    print(f"score : {score}")
    print(f'Total boards explored: {boards_explored}')
    print(f'Time to complete minimax: {toc - tic:0.04f} seconds')

def run_code_tests():
    '''
    b1 : expect win for X (1)  < 200 boards explored
    b1 = np.array([[1, 0, -1], [1, 0, 0], [-1, 0, 0]])

    In addtion to the board b1, run tests on the following
    boards:
       b2:  expect win for O (-1)  > 1000 boards explored
       b2 = np.array([[0, 0, 0], [1, -1, 1], [0, 0, 0]])

       b3: expect TIE (0)  > 500,000 boards explored; time around 20secs
       b3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

       b4: expect TIE(0) > 7,000,000 boards;  time around 4-5 mins
       b4 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, -1], [0, -1, 1, 0], [0, 0, 0, -1]])

    '''
    global boards_explored
    boards_explored = 0
    # Minimax for a board: evaluate the board
    #    expect win for X (1)  < 200 boards explored
    b1 = np.array([[1, 0, -1], [1, 0, 0], [-1, 0, 0]])
    b2 = np.array([[0, 0, 0], [1, -1, 1], [0, 0, 0]])
    b3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    b4 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, -1], [0, -1, 1, 0], [0, 0, 0, -1]])

    # custom test cases to validate minimax
    # nearly complete X win (should come back 1)
    j0 = np.array([[-1, -1, 1],
                   [-1, -1, 1],
                   [1, 0, 0]])
    # nearly complete O win (should come back -1)
    j1 = np.array([[-1, 0, -1],
                   [-1, 1, 1],
                   [0, 1, 1]])
    # pre-determined draw state (should come back 0)
    j2 = np.array([[1, -1, 1],
                   [-1, 1, 1],
                   [-1, 1, -1]])

    test_cases = {'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'j0': j0,
                  'j1': j1, 'j2': j2}

    # tests 1 - 4 are Dr C provided
    # tests 5 - 7 are James C tests to validate minimax
    chosen_test_case = 5  # change this to correspond with 1 for test b1
    counter = 0
    for key, value in test_cases.items():
        counter += 1
        if counter == chosen_test_case:
            test_name = key
            test_board = value

    run_minimax(test_name, test_board)

    # read time before and after call to minimax for b2
    # tic = time.perf_counter()
    # score = minimax(b2, max_depth, is_x_to_move)
    # toc = time.perf_counter()
    # print('TESTING Board b2')
    # print(f"score : {score}")
    # print(f'Total boards explored: {boards_explored}')
    # print(f'Time to complete minimax: {toc - tic:0.04f} seconds')






if __name__ == '__main__':
    run_code_tests()

