import numpy as np


def read_spe_perm(filename, nx=60, ny=220, nz=85):
    print(f"читаем файлик {filename} ...")
    data_2d = np.loadtxt(filename)
    data_1d = data_2d.ravel()
    nxyz = nx * ny * nz
    Kx_1d = data_1d[0: nxyz]
    Ky_1d = data_1d[nxyz: 2 * nxyz]
    Kz_1d = data_1d[2 * nxyz: 3 * nxyz]
    Kx = Kx_1d.reshape((nz, ny, nx), order='C')
    Ky = Ky_1d.reshape((nz, ny, nx), order='C')
    Kz = Kz_1d.reshape((nz, ny, nx), order='C')

    print("размеры:")
    print("Kx:", Kx.shape, "Ky:", Ky.shape, "Kz:", Kz.shape)
    return Kx, Ky, Kz


def write_vtk_3d_structured(filename, Kx, Ky, Kz):
    nz, ny, nx = Kx.shape
    npoints = nx * ny * nz

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("SPE10 permeability data\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")

        f.write(f"POINT_DATA {npoints}\n")

        f.write("SCALARS KX float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{Kx[k, j, i]:.6g}\n")

        f.write("SCALARS KY float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{Ky[k, j, i]:.6g}\n")

        f.write("SCALARS KZ float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{Kz[k, j, i]:.6g}\n")


def write_vtk_2d_slice(filename, Kx, Ky, Kz, k_slice=0):
    nz, ny, nx = Kx.shape
    if not (0 <= k_slice < nz):
        raise ValueError(f"k_slice={k_slice} вне диапазона 0..{nz - 1}")
    npoints_2d = nx * ny

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("SPE10 permeability slice data\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")

        f.write(f"POINT_DATA {npoints_2d}\n")

        f.write("SCALARS KX float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{Kx[k_slice, j, i]:.6g}\n")

        f.write("SCALARS KY float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{Ky[k_slice, j, i]:.6g}\n")

        f.write("SCALARS KZ float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{Kz[k_slice, j, i]:.6g}\n")


def split_layer_to_squares(layer_2d, square_size, output_prefix):
    ny, nx = layer_2d.shape
    count = 0
    for j_start in range(0, ny, square_size):
        for i_start in range(0, nx, square_size):
            square = layer_2d[j_start:j_start + square_size, i_start:i_start + square_size]

            output_filename = f"{output_prefix}_{count}.vtk"
            write_vtk_2d_square(output_filename, square)
            count += 1


def write_vtk_2d_square(filename, square):
    ny, nx = square.shape
    npoints_2d = nx * ny

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("2D permeability square\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")

        f.write(f"POINT_DATA {npoints_2d}\n")
        f.write("SCALARS PERMEABILITY float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for j in range(ny):
            for i in range(nx):
                f.write(f"{square[j, i]:.6g}\n")


nx, ny, nz = 60, 220, 85
input_file = "spe_perm.dat"

Kx, Ky, Kz = read_spe_perm(input_file, nx, ny, nz)

# write_vtk_3d_structured("perm_3d.vtk", Kx, Ky, Kz)

k_slice = 3
# write_vtk_2d_slice("perm_slice_k10.vtk", Kx, Ky, Kz, k_slice=k_slice)

square_size = 20

layer_2d = Kx[k_slice, :, :]

split_layer_to_squares(layer_2d, square_size, output_prefix=f"PERM/slice_k{k_slice}_square")
