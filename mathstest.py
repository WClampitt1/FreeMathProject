from maths import Matrix


# A = Matrix('A', [[1, 20000, 3], [4, 5, 6], [7, 8, 0]])

Matrix.import_matrix('B', 'A.dat')

A = Matrix.import_matrix('A', 'A.dat')

A.display()

B = Matrix.gen_rand_int_matrix('B', 4, 4, 0, 10)
B.display()
