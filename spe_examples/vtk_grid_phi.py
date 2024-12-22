import numpy as np


def write_vtk_structured_points_phi(filename, phi_data):
    if phi_data.ndim == 2:
        ny, nx = phi_data.shape
        nz = 1  # один слой
    elif phi_data.ndim == 3:
        nz, ny, nx = phi_data.shape
    else:
        raise ValueError("phi_data должен быть либо 2D, либо 3D массивом")

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("SPE PHI data\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write("SCALARS PHI float 1\n")
        f.write("LOOKUP_TABLE default\n")
        if phi_data.ndim == 3:
            for z in range(nz):
                for y in range(ny):
                    row = []
                    for x in range(nx):
                        row.append(f"{phi_data[z, y, x]:.6g}")
                    f.write(" ".join(row) + "\n")
        elif phi_data.ndim == 2:
            for y in range(ny):
                row = []
                for x in range(nx):
                    row.append(f"{phi_data[y, x]:.6g}")
                f.write(" ".join(row) + "\n")


def split_layer_into_squares_and_save(layer, square_size, output_prefix):
    ny, nx = layer.shape
    k = square_size

    count = 0
    for j in range(0, ny, k):
        for i in range(0, nx, k):
            square = layer[j:j + k, i:i + k]
            output_filename = f"{output_prefix}_square_{count}.vtk"
            write_vtk_structured_points_phi(output_filename, square)
            count += 1


nx, ny, nz = 60, 220, 85

phi_raw = np.loadtxt("spe_phi.dat")
phi_3d = phi_raw.reshape((nz, ny, nx))  # k, j, i

phi_first_layer = phi_3d[2, :, :]  # управление слоями
# write_vtk_structured_points_phi("spe_phi.vtk", phi_3d)
split_layer_into_squares_and_save(phi_first_layer, 20, "PHI/slice_k2")
