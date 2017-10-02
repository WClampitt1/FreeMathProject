from maths import Matrix

A = Matrix('A', [[1, 2],
                 [4, 9],
                 [7, 8],
                 [10, 11]])

B = Matrix('B', [[1, 2, 3, 4],
                 [0, 4, 0, 6]])

A.display()
B.display()

try:
    C = Matrix.mat_multiply(A, B, 'C')
except IndexError:
    print('Error! Inner dimensions must match!')

C.display()
