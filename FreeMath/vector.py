from FreeMath.matrix import Matrix


# TODO
# Vector class needs cross and dot product methods
class Vector(Matrix):

    @staticmethod
    def dot(a, b, w):
        u = [[]]
        v = []
        for row in a.matrix:
            for col in row:
                u[0].append(col)
        for row in b.matrix:
            for col in row:
                v.append([col])
        q = Vector.build(a.get_name(), u)
        p = Vector.build(b.get_name(), v)
        return Vector.matrix_product(q, p, w, vector=True)