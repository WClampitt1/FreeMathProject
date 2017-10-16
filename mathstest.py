from freemath import Matrix, Vector

A = Matrix.random_integer_matrix('A', 5, 5, 0, 5)
B = Matrix.random_integer_matrix('B', 5, 5, 0, 5)

Matrix.matrix_product(A, B, 'C').display()

Matrix.opt_matrix_product(A, B, [0, 4], [0, 4], '')
