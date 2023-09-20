import numpy as np

def evaluate(board):
    '''returns 1 for X win, -1 for O win, 0 for tie OR game in progress
    Using numpy functions to add values in rows and cols
    If we get a sum equal to size of row,col,diag (plus or minus)
     we have a winner
    '''
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
            win = 1
            if (cols < 0):
                wins = -1
            else:
                wins = 1

    # check for diag win
    diag_sums = []
    down_diag = np.diag(board, k=0)
    up_diag = np.diag(np.fliplr(board))
    diag_sums.append(np.sum(down_diag))
    diag_sums.append(np.sum(up_diag))
    if (row_dim == col_dim):
        if(((abs(np.sum(down_diag))) == row_dim) or
                ((abs(np.sum(up_diag))) == row_dim)):
            if(((np.sum(up_diag) < 0)) or (np.sum(down_diag)) < 0):
                win = -1
            else:
                win = 1

    print(f'Our Board:\n{board}')
    print(f'Row Sums: {row_sums}')
    print(f'Col Sums: {col_sums}')
    print(f'Down Diag {down_diag}')
    print(f'Up Diag {up_diag}')

    return win


#### TEST CODE ##########
def run_tests():
    b = np.array([[1, 0, -1], [1, 0, 0], [-1, 0, 0]])

    #TEST1 : No winner
    b = np.array([[1, 0, -1], [1, 0, 0], [-1, 0, 0]])
    score = evaluate(b)
    expect = 0

    if score == expect:
        print (f"PASS Test 1 No Win")
    else: print (f"FAIL Test 1 No Win: \
        expect: {expect} actual: {score}")

    #TEST 2: Win for X
    b = np.array([[1, 0, -1], [1, 0, 0], [1, 0, -1]])
    score = evaluate(b)
    expect = 1

    if score == expect:
        print (f"PASS Test 2  column win")
    else: print (f"FAIL Test 2: column win \
        expect: {expect} actual: {score}")

    #TEST3 Win for O
    b = np.array([[-1, -1, -1], [1, 0, 1], [1, 0, 1]])
    score = evaluate(b)
    expect = -1

    if score == expect:
        print (f"PASS Test 3  row win")
    else: print (f"FAIL Test 3: row win \
        expect: {expect} actual: {score}")


    #TEST4 Win for X on diagonal
    b = np.array([[-1, -1, 1], [1, 1, 1], [1, 0, -1]])
    score = evaluate(b)
    expect = 1

    if score == expect:
        print (f"PASS Test 4  diag win")
    else: print (f"FAIL Test 4: diag win \
        expect: {expect} actual: {score}")

    #TEST5 win for O on reverse diagonal
    b = np.array([[-1, 1, 1], [1, -1, -1], [1, 0, -1]])
    score = evaluate(b)
    expect = -1

    if score == expect:
        print (f"PASS Test 5  diag2 win")
    else: print (f"FAIL Test 5: diag2 win \
        expect: {expect} actual: {score}")

    #TEST6 win for O on reverse diagonal for 4x4 board
    b = np.array([[-1, 1, 1, 0], [1, -1, -1, 0], \
                  [1, 0, -1, 0], [1,0,0,-1]])
    score = evaluate(b)
    expect = -1

    if score == expect:
        print (f"PASS Test 6  diag2 win 4x4")
    else: print (f"FAIL Test 6: diag2 win 4x4 \
        expect: {expect} actual: {score}")


run_tests()
