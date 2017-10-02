# As general documentation for building a matrix:
# *When declaring the object, the first argument should be the name you used
#  to declare the object as a string:
#  matrix_name = Matrix('matrix_name', A) where A is a properly formatted python list.
#  This is so that the Matrix object itself knows it's own name (for convenience, as
#  demonstrated by the display method, which needs such information for clarity).
# *The Matrix class is really just a wrapper for a python list, used to make matrix
#  operations simpler.
from random import randint


class Matrix:
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = matrix

    def get_name(self):
        return self.name

    def display(self, suppress_output=False):
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
        if not suppress_output:
            print(self.name + ' = (\n' + printable_matrix + '\b\b\b\b)\n')
        else:
            return printable_matrix

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

    @staticmethod
    def add_matrices(A, B, C):
        m, n = A.size()
        p, q = B.size()
        if m != p or n != q:
            raise IndexError('Matrices must be the same size!')
        new_mat = []
        for row in range(m):
            new_mat.append([])
            for col in range(n):
                new_mat[row].append(A.get_elem(row+1, col+1) + B.get_elem(row+1, col+1))
        return Matrix('C', new_mat)

    @staticmethod
    def import_matrix(new_mat_name, file_name):
        mat = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                mat.append(line.split())
        for m in range(len(mat)):
            for n in range(len(mat[m])):
                mat[m][n] = int(mat[m][n])
        return Matrix(new_mat_name, mat)

    # A, B are matrices; C is a string, naming A*B. It should be the same name
    # as a string that the programmer wants the matrix to be named in his program.
    # For instance:
    # my_matrix = Matrix.mat_multiply(A, B, 'my_matrix') where A and B are Matrix
    # objects.
    # function returns C as a matrix object
    @staticmethod
    def mat_multiply(A, B, C, show_percent_complete=False):  # matrix multiplication
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

        return Matrix(C, new_matrix)

    @staticmethod
    def scalar_multiply(A, c, B):
        m, n = A.size()
        new_mat = []
        for row in range(m):
            new_mat.append([])
            for col in range(n):
                new_mat[row].append(A.get_elem(row+1, col+1) * c)
        return Matrix(B, new_mat)

    # get a matrix of predefined size of random integers
    @staticmethod
    def gen_rand_int_matrix(matrix_name, rows, columns, rand_min, rand_max):
        mat = []
        for i in range(rows):
            mat.append([])
            for j in range(columns):
                mat[i].append(randint(rand_min, rand_max))
        return Matrix(matrix_name, mat)


class Vector(Matrix):
    def test(self):
        pass




