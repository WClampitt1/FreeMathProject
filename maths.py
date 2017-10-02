# As general documentation for building a matrix:
# *When declaring the object, the first argument should be the name you used
#  to declare the object as a string:
#  matrix_name = Matrix('matrix_name', A) where A is a properly formatted python list.
#  This is so that the Matrix object itself knows it's own name (for convenience, as
#  demonstrated by the display method, which needs such information for clarity).
# *The Matrix class is really just a wrapper for a python list, used to make matrix
#  operations simpler.


class Matrix:
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = matrix

    def get_name(self):
        return self.name

    def display(self, outer_call=True):
        largest_elem_len = 1
        printable_matrix = '\t'
        for row in self.matrix:
            for col in row:
                if len(str(col)) > largest_elem_len:
                    largest_elem_len = len(str(col))
        for i in self.matrix:
            for k in i:
                printable_matrix += (str(k) + '  ' + (largest_elem_len - len(str(k))) * ' ')
            printable_matrix += '\n\t'
        if outer_call:
            print(self.name + ' = (\n' + printable_matrix + '\b\b\b\b)\n')
        else:
            return printable_matrix

    def size(self):
        return len(self.matrix), len(self.matrix[0])

    def get_elem(self, i, j):
        return self.matrix[i-1][j-1]

    def write_out(self, file_name):
        writable_matrix = self.display(False)
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(writable_matrix)

    # A, B are matrices; C is a string, naming A*B. It should be the same name
    # as a string that the programmer wants the matrix to be named in his program.
    # For instance:
    # my_matrix = Matrix.mat_multiply(A, B, 'my_matrix') where A and B are Matrix
    # objects.
    # function returns C as a matrix object
    @staticmethod
    def mat_multiply(A, B, C):  # matrix multiplication
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

        return Matrix(C, new_matrix)





