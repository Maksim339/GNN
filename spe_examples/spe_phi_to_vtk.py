import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata


def generate_points_with_density(nx, ny, spacing, center, num_points):
    x_max, y_max = nx * spacing, ny * spacing
    max_radius = np.sqrt((x_max / 2)**2 + (y_max / 2)**2)

    points = []
    while len(points) < num_points:
        x = np.random.uniform(0, x_max)
        y = np.random.uniform(0, y_max)

        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        density_factor = (1 - r / max_radius)**2
        if np.random.rand() < density_factor:
            points.append([x, y])

    return np.array(points)


def generate_adaptive_unstructured_grid(phi_layer, spacing, output_file):
    ny, nx = phi_layer.shape
    x = np.linspace(0, nx * spacing, nx)
    y = np.linspace(0, ny * spacing, ny)
    xv, yv = np.meshgrid(x, y)

    center = (nx * spacing / 2, ny * spacing / 2)

    num_points = nx * ny * 3
    random_points = generate_points_with_density(nx, ny, spacing, center, num_points)

    points = np.vstack((random_points, np.column_stack((xv.ravel(), yv.ravel()))))

    delaunay = Delaunay(points)
    triangles = delaunay.simplices

    phi_values = griddata((xv.ravel(), yv.ravel()), phi_layer.ravel(), points, method='linear', fill_value=0)

    with open(output_file, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Adaptive Unstructured Grid\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {len(points)} float\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} 0\n")

        f.write(f"CELLS {len(triangles)} {len(triangles) * 4}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

        f.write(f"CELL_TYPES {len(triangles)}\n")
        f.write("5\n" * len(triangles))

        f.write(f"POINT_DATA {len(points)}\n")
        f.write("SCALARS PHI float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for phi in phi_values:
            f.write(f"{phi}\n")


def generate_mixed_grid_with_rect_tri(phi_layer, spacing, output_file):
    ny, nx = phi_layer.shape
    x = np.linspace(0, nx * spacing, nx)
    y = np.linspace(0, ny * spacing, ny)
    xv, yv = np.meshgrid(x, y)

    margin = min(nx, ny) // 4
    center_points = []
    boundary_rectangles = []

    for j in range(ny - 1):
        for i in range(nx - 1):
            if i < margin or i >= nx - margin or j < margin or j >= ny - margin:
                p1 = j * nx + i
                p2 = j * nx + (i + 1)
                p3 = (j + 1) * nx + (i + 1)
                p4 = (j + 1) * nx + i
                boundary_rectangles.append([p1, p2, p3, p4])
            else:
                center_points.append([xv[j, i], yv[j, i]])

    center_points = np.array(center_points)
    delaunay = Delaunay(center_points)
    triangles = delaunay.simplices

    phi_values = griddata((xv.ravel(), yv.ravel()), phi_layer.ravel(), center_points, method='linear', fill_value=0)

    with open(output_file, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Mixed Grid: Rectangular (Boundary) and Triangular (Center)\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        points = np.vstack((np.column_stack((xv.ravel(), yv.ravel())), center_points))
        f.write(f"POINTS {len(points)} float\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} 0\n")

        total_cells = len(boundary_rectangles) + len(triangles)
        f.write(f"CELLS {total_cells} {total_cells * 5 + len(triangles)}\n")
        for rect in boundary_rectangles:
            f.write(f"4 {rect[0]} {rect[1]} {rect[2]} {rect[3]}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

        f.write(f"CELL_TYPES {total_cells}\n")
        f.write("9\n" * len(boundary_rectangles))
        f.write("5\n" * len(triangles))

        f.write(f"POINT_DATA {len(points)}\n")
        f.write("SCALARS PHI float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for phi in phi_layer.ravel():
            f.write(f"{phi}\n")
        for phi in phi_values:
            f.write(f"{phi}\n")



nx, ny, nz = 60, 220, 85
phi_raw = np.loadtxt("spe_phi.dat")
phi_3d = phi_raw.reshape((nz, ny, nx))
phi_first_layer = phi_3d[0, :, :]
# generate_adaptive_unstructured_grid(phi_first_layer, spacing=1.0, output_file="adaptive_center_dense_phi_layer.vtk")
generate_mixed_grid_with_rect_tri(phi_first_layer, spacing=1.0, output_file="mixed_grid_rect_tri_phi_layer.vtk")
