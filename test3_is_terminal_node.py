import numpy as np

# test is_terminal_node
# tip: use your tested evaluate() to help with this


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

    print(f'\nOur Board:\n{board}')
    print(f'Row Sums: {row_sums}')
    print(f'Col Sums: {col_sums}')
    print(f'Diag Sums: {diag_sums}')

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

    return terminal_board


#### TEST CODE ##########
def run_tests():

    # TEST1 : Not terminal
    b = np.array([[1, 0, -1], [1, 0, 0], [-1, 0, 0]])
    is_terminal = is_terminal_node(b)
    expected = False

    if is_terminal == expected:
        print(f"PASS Test 1 Non Terminal Board")
    else:
        print(f"FAIL Test 1 Non Terminal Board: \
        expect: {expected} actual: {is_terminal}")

    # TEST 2: Terminal
    b = np.array([[1, 1, 1], [1, -1, -1], [-1, 0, 0]])
    is_terminal = is_terminal_node(b)
    expected = True

    if is_terminal == expected:
        print(f"PASS Test 2  Terminal Board")
    else:
        print(f"FAIL Test 2  Terminal Board: \
            expect: {expected} actual: {is_terminal}")

    # TEST3
    b = np.array([[1, -1, 1], [1, 1, -1], [-1, 1, -1]])
    is_terminal = is_terminal_node(b)
    expected = True

    if is_terminal == expected:
        print(f"PASS Test 3  Terminal Board")
    else:
        print(f"FAIL Test 3  Terminal Board: \
            expect: {expected} actual: {is_terminal}")

    # TEST4 Win for X on diagonal
    b = np.array([[1, 0, 0], [0, 1, -1], [-1, -1, 1]])
    is_terminal = is_terminal_node(b)
    expected = True

    if is_terminal == expected:
        print(f"PASS Test 4  Terminal Board")
    else:
        print(f"FAIL Test 4  Terminal Board: \
            expect: {expected} actual: {is_terminal}")

    # TEST5 win for O on reverse diagonal
    b = np.array([[1, 1, -1], [0, -1, 1], [-1, -1, 1]])
    is_terminal = is_terminal_node(b)
    expected = True

    if is_terminal == expected:
        print(f"PASS Test 5  Terminal Board")
    else:
        print(f"FAIL Test 5  Terminal Board: \
            expect: {expected} actual: {is_terminal}")

    # TEST6 win for O on reverse diagonal for 4x4 board
    b = np.array([[1, 1, 0, -1], [0, 0, -1, 1],
                  [0, -1, 1, 0], [-1, 0, 0, 0]])
    is_terminal = is_terminal_node(b)
    expected = True

    if is_terminal == expected:
        print(f"PASS Test 6  Terminal Board")
    else:
        print(f"FAIL Test 6  Terminal Board: \
                expect: {expected} actual: {is_terminal}")


run_tests()
