from maths import Matrix, Vector

# this is a comment

A = Matrix('A', [[1, 2, 3],
                 [4, 5, 6],
                 [8, 9, 10]])
A.display()

x = Vector('x', [[1],
                 [2],
                 [3]])
x.display()
b = Matrix.matrix_product(A, x, 'b')
b.display()
