# MiniMax - Get score for board

import math
import numpy as np
import time
import copy
from queue import PriorityQueue

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


def win(board, x_or_o):
    """ Algorithm that detects win/block states for the player and the
    opponent for the current board.
    :param board: a 2d numpy array (should be a square N x N matrix)
    :param x_or_o: char X or O
    return wins (bool) and blocks (bool) that are detect the presence of
    an immediate winning or blocking state.
    """
    wins = False

    # check player
    if(x_or_o == 'X'):
        x_or_o = 1
    else:
        x_or_o = -1

    # get row_sum, col_sum, and diag_sums
    row_sums = np.sum(board, axis=1)
    col_sums = np.sum(board, axis=0)

    diag_sums = []
    down_diag = np.diag(board, k=0)
    up_diag = np.diag(np.fliplr(board))
    diag_sums.append(np.sum(down_diag))
    diag_sums.append(np.sum(up_diag))

    # check for threats (2 spaces filled + empty) the player presents

    for rows in row_sums:
        if rows == (2 * x_or_o):
            wins = True

    if (wins == False):
        for cols in col_sums:
            if cols == (2 * x_or_o):
                wins = True
                break

    if(wins == False):
        for diags in diag_sums:
            if diags == (2 * x_or_o):
                wins = True
                break

    return wins


def block(board, x_or_o):
    """ Algorithm that detects win/block states for the player and the
    opponent for the current board.
    :param board: a 2d numpy array (should be a square N x N matrix)
    :param x_or_o: char X or O
    return wins (bool) and blocks (bool) that are detect the presence of
    an immediate winning or blocking state.
    """
    blocks = False

    # check player
    if(x_or_o == 'X'):
        x_or_o = 1
    else:
        x_or_o = -1

    # get row_sum, col_sum, and diag_sums
    row_sums = np.sum(board, axis=1)
    col_sums = np.sum(board, axis=0)

    diag_sums = []
    down_diag = np.diag(board, k=0)
    up_diag = np.diag(np.fliplr(board))
    diag_sums.append(np.sum(down_diag))
    diag_sums.append(np.sum(up_diag))

    # check for threats (2 spaces filled + empty) the opponent presents
    for rows in row_sums:
        if rows == (-2 * x_or_o):
            blocks = True
            break

    if(blocks == False):
        for cols in col_sums:
            if cols == (-2 * x_or_o):
                blocks = True
                break

    if (blocks == False):
        for diags in diag_sums:
            if diags == (-2 * x_or_o):
                blocks = True
                break

    return blocks


def center(board):
    """Look for center move open
    :param board: a 2d numpy array (should be a square N x N matrix)
    return (bool) center that determines if the board center is
    available for play. For even grid matrix, considers bottom right of
    4 square 'center' the center.
    """
    center = False
    row_dim, col_dim = board.shape

    # determine if center space is open
    if(board[row_dim // 2][col_dim // 2] == 0):
        center = True

    return center


def corner(board):
    """Look for center move open
    :param board: a 2d numpy array (should be a square N x N matrix)
    return (bool) corner that determines if the board corners are
    available for play.
    """
    corner = False
    row_dim, col_dim = board.shape

    # check for corner cases
    if(board[0][0] == 0):  # top left
        corner = True
    elif(board[0][col_dim - 1] == 0):  # top right
        corner = True
    elif(board[row_dim - 1][0] == 0):  # bottom left
        corner = True
    elif(board[row_dim - 1][col_dim - 1] == 0):  # bottom right
        corner = True

    return corner


def priority_child_list(children, player):
    """Defines move priority
    :param children: a list of numpy 2d children arrays.
    :param player: char X or O
    returns list ordered_list
    """
    priority_list = PriorityQueue()
    ordered_list = []

    for count, child in enumerate(children):
        # look for wins - top priority, priority 0
        found_win = win(child, player)
        if(found_win):
            priority_list.put((0, count, child))
            continue

        # look for block - second priority, priority 1
        found_block = block(child, player)
        if(found_block):
            priority_list.put((1, count, child))
            continue

        # look for center - third priority, priority 2
        found_center = center(child)
        if(found_center):
            priority_list.put((2, count, child))
            continue

        # look for corner - fourth priority, priority 3
        found_corner = corner(child)
        if(found_corner):
            priority_list.put((3, count, child))
            continue

        # all else - last priority, priority 4
        else:
            priority_list.put((4, count, child))

    for items in range(priority_list.qsize()):
        priority, random, data = priority_list.get()
        ordered_list.append(data)

    return ordered_list


def minimax_ab(board, depth, alpha, beta, maximizing_player):
    """
       0 (draw) 1 (win for X) -1 (win for O)
       Explores all child boards for this position and returns
       the best score given that all players play optimally
       :param board: a 2d numpy array (should be a square N x N matrix)
       :param depth: int provides the current depth as # remaining moves
       :param alpha: int passed in as -Inf to use in Alpha/Beta Pruning
       :param beta: int passed in as +Inf to use in Alpha/Beta Pruning
       :param maximizing_player: bool -> False = O turn; True = X turn
       returns: the value of the board
    """

    global boards_explored
    if depth == 0 or is_terminal_node(board):
        if(depth == 0):
            boards_explored += 1
        return evaluate(board)

    if maximizing_player:  # max player plays X
        max_eva = -math.inf
        print('For X Turn')
        child_list = get_child_boards(board, 'X')
        priority_children = priority_child_list(child_list, 'X')
        for child_board in priority_children:
            eva = minimax_ab(child_board, depth-1, alpha, beta, False)
            max_eva = max(max_eva, eva)
            alpha = max(alpha, max_eva)

            if alpha > beta:  # max quits if Alpha > Beta
                break

        return max_eva

    else:             # minimizing player
        min_eva = math.inf
        print('For O Turn')
        child_list = get_child_boards(board, 'O')
        priority_children = priority_child_list(child_list, 'O')
        for child_board in priority_children:
            eva = minimax_ab(child_board, depth - 1, alpha, beta,  True)
            min_eva = min(min_eva, eva)
            beta = min(beta, min_eva)

            if beta <= alpha:  # min quits when Beta <= Alpha
                break

        return min_eva


def run_minimax_ab(board_name, board):
    """
    Function designed to call minimax with alpha-beta pruning.
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

    # Set Alpha Beta to default values
    alpha = -math.inf
    beta = math.inf

    # read time before and after call to minimax for b1
    tic = time.perf_counter()
    score = minimax_ab(board, max_depth, alpha, beta, is_x_to_move)
    toc = time.perf_counter()
    print(f'TESTING Board {board_name}')
    print(f"score : {score}")
    print(f'Total boards explored: {boards_explored}')
    print(f'Time to complete minimax: {toc - tic:0.04f} seconds')


def run_code_tests():
    """
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

    """
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
    chosen_test_case = 2  # change this to correspond with 1 for test b1
    counter = 0
    for key, value in test_cases.items():
        counter += 1
        if counter == chosen_test_case:
            test_name = key
            test_board = value

    run_minimax_ab(test_name, test_board)


if __name__ == '__main__':
    run_code_tests()
