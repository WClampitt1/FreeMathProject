from freemath import Matrix, Vector

B = [[-1, 1, -1, 1, 1], [0, 0, 0, 1, 1], [1, 1, 1, 1, 3], [8, 4, 2, 1, 1]]
R = B
Bdawg = Matrix.row_reduce(R)
M = Matrix('M', B)
N = Matrix('N', Bdawg)
M.display()
N.display()
