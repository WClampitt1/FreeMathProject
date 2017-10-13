from freemath import Matrix


def row_reduce(A):
    A_prime = A
    not_reduced = True
    m, n = 0, 0
    count = 0
    while not_reduced:
        if A_prime[m][m] is 0:
            A_prime.append(A_prime.pop(m))
            count += 1
        elif A_prime[m][m] is 1:
            for i in len(A_prime)-1:
                if i is not m:
                    A_prime[i] = [-A_prime[m][m] + x for x in A_prime[i]]
            m += 1
        else:
            # A_prime[m] = [x / A_prime[m][m] for x in A_prime[m]]
            for i in range(len(A_prime[m])):
                A_prime[m][i] = A_prime[m][i]/A_prime[m][m]

        break

    return A_prime


B = [[-1, 1, -1, 1, 1], [0, 0, 0, 1, 1], [1, 1, 1, 1, 3], [8, 4, 2, 1, 1]]
R = B
Bdawg = row_reduce(R)
M = Matrix('M', B)
N = Matrix('N', Bdawg)
M.display()
N.display()
