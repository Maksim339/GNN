import numpy as np

N = 10
k = 1.0
p0 = 0.0  # ГУ
delta_x = 1.0


def solve_pressure_field(N, k, p0, delta_x, injection_pos, production_pos, injection_strength=1.0, production_strength=-1.0):
    total_cells = N * N
    A = np.zeros((total_cells, total_cells))
    q = np.zeros(total_cells)
    q_i_array = np.zeros(total_cells)

    injection_idx = injection_pos[0] * N + injection_pos[1]
    production_idx = production_pos[0] * N + production_pos[1]
    q_i_array[injection_idx] = injection_strength
    q_i_array[production_idx] = production_strength

    for i in range(N):
        for j in range(N):
            idx = i * N + j

            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                A[idx, idx] = 1.0
                q[idx] = p0
            else:
                neighbors = []
                if j - 1 >= 0:
                    neighbors.append(((i, j - 1), -1.0))
                if j + 1 < N:
                    neighbors.append(((i, j + 1), -1.0))
                if i - 1 >= 0:
                    neighbors.append(((i - 1, j), -1.0))
                if i + 1 < N:
                    neighbors.append(((i + 1, j), -1.0))

                A[idx, idx] = len(neighbors)

                for neighbor, coeff in neighbors:
                    n_i, n_j = neighbor
                    n_idx = n_i * N + n_j
                    A[idx, n_idx] = coeff

                q_i = q_i_array[idx]
                q[idx] = -(delta_x ** 2 / k) * q_i

    p = np.linalg.solve(A, q)
    return p.reshape((N, N))
