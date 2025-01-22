import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def read_permeability_from_vtk(filename):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    dims = data.GetDimensions()
    nx, ny, nz = dims[0], dims[1], dims[2]
    if nz != 1:
        raise ValueError("nz != 1")

    vtk_array = data.GetPointData().GetScalars()
    if not vtk_array:
        raise ValueError("не найден скалярный массив в VTK")

    k_flat = vtk_to_numpy(vtk_array)
    k_2d = k_flat.reshape((ny, nx), order='C')

    return k_2d


def write_pressure_to_vtk(filename, pressure_array, spacing=1.0):
    ny, nx = pressure_array.shape

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, 1)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.SetSpacing(spacing, spacing, 1.0)

    pressure_flat = pressure_array.ravel(order='C')
    vtk_arr = numpy_to_vtk(pressure_flat, deep=True, array_type=vtk.VTK_DOUBLE)
    vtk_arr.SetName("PRESSURE")

    image.GetPointData().SetScalars(vtk_arr)

    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image)
    writer.Write()


def build_rhs_q_array(N, injection_pos, production_pos, injection_strength=1.0, production_strength=-1.0):
    q_arr = np.zeros((N, N), dtype=float)
    q_arr[injection_pos[0], injection_pos[1]] = injection_strength
    q_arr[production_pos[0], production_pos[1]] = production_strength
    return q_arr


def solve_pressure_field(
        k_array,
        q_array,
        delta_x=1.0,
        production_pos=(0, 0)
):
    N = k_array.shape[0]

    total_cells = N * N
    A = np.zeros((total_cells, total_cells), dtype=float)
    b = np.zeros(total_cells, dtype=float)

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            center_index = idx(i, j)

            if (i, j) == production_pos:
                A[center_index, center_index] = 1.0
                b[center_index] = 0.0
                continue

            if j > 0:
                k_w = 2.0 * k_array[i, j] * k_array[i, j - 1] / (k_array[i, j] + k_array[i, j - 1])
            else:
                k_w = 0.0
            if j < N - 1:
                k_e = 2.0 * k_array[i, j] * k_array[i, j + 1] / (k_array[i, j] + k_array[i, j + 1])
            else:
                k_e = 0.0
            if i > 0:
                k_n = 2.0 * k_array[i, j] * k_array[i - 1, j] / (k_array[i, j] + k_array[i - 1, j])
            else:
                k_n = 0.0
            if i < N - 1:
                k_s = 2.0 * k_array[i, j] * k_array[i + 1, j] / (k_array[i, j] + k_array[i + 1, j])
            else:
                k_s = 0.0
            A[center_index, center_index] = (k_w + k_e + k_n + k_s)

            if j > 0:
                A[center_index, idx(i, j - 1)] = -k_w
            if j < N - 1:
                A[center_index, idx(i, j + 1)] = -k_e
            if i > 0:
                A[center_index, idx(i - 1, j)] = -k_n
            if i < N - 1:
                A[center_index, idx(i + 1, j)] = -k_s

            b[center_index] = q_array[i, j] * (delta_x ** 2)

    p_vector = np.linalg.solve(A, b)

    p_field = p_vector.reshape((N, N))
    return p_field


# filename_k = "spe_examples/PERM/slice_k0_square_1.vtk"  # наша проницаемость на сетке
#
# k_2d = read_permeability_from_vtk(filename_k)
# N = k_2d.shape[0]
#
# # =====координаты=====
# injection = (5, 5)
# production = (15, 15)
# q_2d = build_rhs_q_array(N, injection, production, 1.0, -1.0)
#
# p_result = solve_pressure_field(
#     k_array=k_2d,
#     q_array=q_2d,
#     delta_x=0.01,
#     production_pos=production
# )
#
# out_vtk = "pressure_result.vtk"
# write_pressure_to_vtk(out_vtk, p_result, spacing=1.0)