# As general documentation for building a matrix:
# *When declaring the object, the first argument should be the name you used
#  to declare the object as a string:
#  matrix_name = Matrix('matrix_name', A) where A is a properly formatted python list.
#  This is so that the Matrix object itself knows it's own name (for convenience, as
#  demonstrated by the display method, which needs such information for clarity).
# *The Matrix class is really just a wrapper for a python list, used to make matrix
#  operations simpler.
from random import randint
from copy import deepcopy


class MatrixOperations:
    @staticmethod
    def add_matrices(A, B, name=''):
        m, n = A.size()
        p, q = B.size()
        if m != p or n != q:
            raise IndexError('Matrices must be the same size!')
        new_mat = []
        for row in range(m):
            new_mat.append([])
            for col in range(n):
                new_mat[row].append(A.get_elem(row+1, col+1) + B.get_elem(row+1, col+1))
        return Matrix.build(name, new_mat)

    @staticmethod
    def import_matrix(file_name, new_mat_name=''):
        mat = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                mat.append(line.split())
        for m in range(len(mat)):
            for n in range(len(mat[m])):
                mat[m][n] = int(mat[m][n])
        return Matrix.build(new_mat_name, mat)

    # A, B are matrices; C is a string, naming A*B. It should be the same name
    # as a string that the programmer wants the matrix to be named in his program.
    # For instance:
    # my_matrix = Matrix.mat_multiply(A, B, 'my_matrix') where A and B are Matrix
    # objects.
    # function returns C as a matrix object
    @staticmethod
    def matrix_product(A, B, C='', show_percent_complete=False, vector=False):  # matrix multiplication
        m, n = A.size()
        p, q = B.size()
        new_matrix = []
        if n != p:  # to assure no dimensional issues
            raise IndexError('Inner dimensions must match!')

        # (1) and (2) vary over the outer dimensions of A and B. These are the
        # dimensions of the product.
        # (3) varies over the inner dimensions, to correlate the individual elements
        # being multiplied.
        for rows in range(m):  # (1)
            new_matrix.append([])
            for elements in range(q):  # (2)
                tmp = 0
                i, j = 0, 0
                while i < n and j < p:  # (3)
                    # the '+1' is because the get_elem method operates on the
                    # human convention of using 1 as the starting index
                    # for mathematical matrices
                    tmp += A.get_elem(rows+1, j+1) * B.get_elem(i+1, elements+1)
                    i += 1
                    j += 1
                new_matrix[rows].append(tmp)
            if show_percent_complete:
                print((rows/m)*100)

        if vector:
            return new_matrix[0][0]
        else:
            return Matrix.build(C, new_matrix)

    @staticmethod
    def scalar_product(A, c, B=''):
        m, n = A.size()
        new_mat = []
        for row in range(m):
            new_mat.append([])
            for col in range(n):
                new_mat[row].append(A.get_elem(row+1, col+1) * c)
        return Matrix.build(B, new_mat)

    # get a matrix of predefined size of random integers
    @staticmethod
    def random_integer_matrix(matrix_name, rows, columns, rand_min, rand_max):
        mat = []
        for i in range(rows):
            mat.append([])
            for j in range(columns):
                mat[i].append(randint(rand_min, rand_max))
        return Matrix.build(matrix_name, mat)

    # this is to keep consistency among object types, while still being able to let vectors
    # inherit Matrix
    @staticmethod
    def build(name, matrix):
        if len(matrix) == 1 or len(matrix[0]) == 1:
            return Vector(name, matrix)
        else:
            return Matrix(name, matrix)


# TODO
# * needs to be able to return echelon form and reduced echelon forms
#   I think that ef() and ref() should be non-static, so it should take
#   the name of the new matrix, and return the ef or ref of the matrix
#   instance calling it. So for some nxm matrix A,
#       B = A.ref('B') or B = A.ef('B')
class Matrix(MatrixOperations):
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = matrix

    def get_name(self):
        return self.name

    def display(self, suppress_output=False, show_label=True):
        largest_elem_len = 1
        # All of the `if not suppress_output:` business basically
        # strips all of the print formatting from the string, so that it easier to work with
        # for mechanical purposes. It leaves the correct spacing, but removes tabs and name printings
        if not suppress_output:
            printable_matrix = '\t'
        else:
            printable_matrix = ''
        for row in self.matrix:
            for col in row:
                if len(str(col)) > largest_elem_len:
                    largest_elem_len = len(str(col))
        for i in self.matrix:
            for k in i:
                printable_matrix += (str(k) + '  ' + (largest_elem_len - len(str(k))) * ' ')
            if not suppress_output:
                printable_matrix += '\n\t'
            else:
                printable_matrix += '\n'
        if not suppress_output and show_label:
            print(self.name + ' = (\n' + printable_matrix + '\b\b\b\b)\n')
        elif not suppress_output and not show_label:
            print(printable_matrix)
        else:
            return printable_matrix

    # returns the row echelon form of the matrix
    def ref(self):
        A = deepcopy(self.matrix)
        row, col, count = 0, 0, 0
        while row != len(A) and col != len(A[row]):
            if A[row][col] == 0:
                A.append(A.pop(row))
                if count == len(A) - 1:
                    col += 1
                    count = 0
                else:
                    count += 1
            elif A[row][col] == 1:
                x = row + 1
                while x < len(A):
                    row_op = [-A[x][col] * i for i in A[row]]
                    y = 0
                    while y < len(A[x]):
                        A[x][y] += row_op[y]
                        y += 1
                    x += 1
                col += 1
                row += 1
            else:
                A[row] = [x / A[row][col] for x in A[row]]

        new_matrix = Matrix(self.name, A)
        return new_matrix

    def size(self):
        return len(self.matrix), len(self.matrix[0])

    # any internal, mechanical use of get_elem will likely need to send (row+1, column+1) to
    # get actual elements, because the program is designed to operate on human use of indexing matrices
    # at 1, 1 not 0, 0
    def get_elem(self, i, j):
        return self.matrix[i-1][j-1]

    def write_out(self, file_name=''):  # defaults to writing matrix to it's own file based on its name
        if file_name == '':
            file_name = self.name + '.dat'
        writable_matrix = self.display(suppress_output=True)
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(writable_matrix)

    def power(self, num, name=''):
        tmp = self
        for i in range(num - 1):
            tmp = Matrix.matrix_product(tmp, self, name)  # order shouldn't matter
        return tmp

    def get_list(self):
        return self.matrix


