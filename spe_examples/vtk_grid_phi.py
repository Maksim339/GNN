import numpy as np


def write_vtk_structured_points_phi(filename, phi_3d):
    nz, ny, nx = phi_3d.shape
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
        for z in range(nz):
            for y in range(ny):
                row = []
                for x in range(nx):
                    row.append(f"{phi_3d[z, y, x]:.6g}")
                f.write(" ".join(row) + "\n")


nx, ny, nz = 60, 220, 85

phi_raw = np.loadtxt("spe_phi.dat")
if phi_raw.size != nx * ny * nz:
    raise ValueError("spe_phi.dat: неверное количество чисел.")
phi_3d = phi_raw.reshape((nz, ny, nx))  # k, j, i

write_vtk_structured_points_phi("spe_phi.vtk", phi_3d)
