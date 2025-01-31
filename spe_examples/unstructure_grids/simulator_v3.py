import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from collections import defaultdict


def rel_perm(Sw):
    krw = Sw**2
    kro = (1.0 - Sw) ** 2
    return krw, kro


def build_cell_data(points, triangles, k_vertex, phi_vertex):
    M = len(triangles)
    cell_centers = np.zeros((M, 2))
    k_cells = np.zeros(M)
    phi_cells = np.zeros(M)

    for i, tri in enumerate(triangles):
        v0, v1, v2 = tri
        k_harm = 3 / (1 / k_vertex[v0] + 1 / k_vertex[v1] + 1 / k_vertex[v2])
        k_cells[i] = k_harm

        phi_cells[i] = (phi_vertex[v0] + phi_vertex[v1] + phi_vertex[v2]) / 3
        cell_centers[i] = points[tri, :2].mean(axis=0)

    return cell_centers, k_cells, phi_cells


def build_connectivity(points, cell_centers, triangles):
    edge_dict = defaultdict(list)
    face_info = [[] for _ in range(len(triangles))]

    for i, tri in enumerate(triangles):
        for j in range(3):
            v1, v2 = sorted([tri[j], tri[(j + 1) % 3]])
            edge_dict[(v1, v2)].append(i)

    for edge, cells in edge_dict.items():
        if len(cells) == 2:
            i, j = cells

            p1 = points[edge[0], :2]
            p2 = points[edge[1], :2]

            edge_vec = p2 - p1
            normal = np.array([edge_vec[1], -edge_vec[0]])
            norm_length = np.linalg.norm(normal)
            if norm_length == 0:
                continue
            normal /= norm_length

            center_i = cell_centers[i]
            face_center = 0.5 * (p1 + p2)
            if np.dot(normal, face_center - center_i) > 0:
                normal = -normal

            face_info[i].append((j, normal))
            face_info[j].append((i, -normal))

    return face_info


def assemble_pressure_system(
    cell_centers, face_info, k_cells, phi, Sw, mu_w, mu_o, wells, dt
):
    M = len(cell_centers)
    A = sp.lil_matrix((M, M))
    b = np.zeros(M)

    krw, kro = rel_perm(Sw)
    lambda_t = krw / mu_w + kro / mu_o

    for i in range(M):
        for j, n in face_info[i]:
            dx = cell_centers[j] - cell_centers[i]
            dist = np.linalg.norm(dx)
            if dist == 0:
                continue

            lambda_avg = (
                2 * lambda_t[i] * lambda_t[j] / (lambda_t[i] + lambda_t[j] + 1e-12)
            )
            T = lambda_avg / dist

            flux_coeff = T * np.dot(n, dx / dist)

            A[i, i] += flux_coeff
            A[i, j] -= flux_coeff

        if i == wells["inj"]["cell"]:
            b[i] += wells["inj"]["rate"]
        if i == wells["prod"]["cell"]:
            b[i] += wells["prod"]["rate"]

    return A.tocsr(), b


def solve_transport(cell_centers, face_info, phi, Sw, p, mu_w, dt):
    M = len(cell_centers)
    Sw_new = Sw.copy()
    krw, kro = rel_perm(Sw)
    lambda_w = krw / mu_w

    for i in range(M):
        flux = 0.0
        for j, n in face_info[i]:
            dp = p[i] - p[j]
            if dp >= 0:
                kr = krw[i]
                lambda_up = lambda_w[i]
            else:
                kr = krw[j]
                lambda_up = lambda_w[j]

            dx = cell_centers[j] - cell_centers[i]
            dist = np.linalg.norm(dx)
            if dist == 0:
                continue

            lambda_avg = 2 * lambda_up * lambda_w[j] / (lambda_up + lambda_w[j] + 1e-12)
            T = lambda_avg / dist

            flux += T * dp

        Sw_new[i] = Sw[i] + (dt / phi[i]) * flux

    return np.clip(Sw_new, 0.0, 1.0)


def interpolate_to_vertices(points, triangles, cell_data):
    vertex_data = np.zeros(len(points))
    vertex_counts = np.zeros(len(points))

    for i, tri in enumerate(triangles):
        for v in tri:
            vertex_data[v] += cell_data[i]
            vertex_counts[v] += 1

    return vertex_data / np.maximum(vertex_counts, 1)


def plot_results(points, triangles, cell_pressure, cell_saturation):
    pressure_vertices = interpolate_to_vertices(points, triangles, cell_pressure)
    saturation_vertices = interpolate_to_vertices(points, triangles, cell_saturation)

    tri = Triangulation(points[:, 0], points[:, 1], triangles)

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    tcf = plt.tricontourf(tri, pressure_vertices, levels=20, cmap="viridis")
    plt.colorbar(tcf, label="P")
    plt.tricontour(tri, pressure_vertices, colors="k", linewidths=0.5, levels=10)
    plt.title("Распределение давления")

    plt.subplot(122)
    tcf = plt.tricontourf(tri, saturation_vertices, levels=20, cmap="Blues")
    plt.colorbar(tcf, label="S_w")
    plt.tricontour(tri, saturation_vertices, colors="k", linewidths=0.5, levels=10)
    plt.title("Распределение насыщенности")

    plt.tight_layout()
    plt.show()


mesh_perm = meshio.read("mesh_perm.vtk")
mesh_phi = meshio.read("mesh_phi.vtk")
points = mesh_perm.points[:, :2]
triangles = None
for cell_block in mesh_perm.cells:
    if cell_block.type == "triangle":
        triangles = cell_block.data
        break

if triangles is None:
    raise ValueError("Треугольные элементы не найдены!")

# ---------------------------------------------------------------------------
mu_w = 0.001
mu_o = 1.0
dt = 0.01
total_time = 5.0
num_steps = int(total_time / dt)

wells = {"inj": {"cell": 233, "rate": 1e-3}, "prod": {"cell": 114, "rate": -1e-3}}

cell_centers, k_cells, phi_cells = build_cell_data(
    points, triangles, mesh_perm.point_data["PERMEABILITY"], mesh_phi.point_data["PHI"]
)

face_info = build_connectivity(points, cell_centers, triangles)

Sw = np.zeros(len(cell_centers))
Sw[:] = 0.2
Sw[wells["inj"]["cell"]] = 1.0
p = np.zeros(len(cell_centers))

for step in range(num_steps):
    A_p, b_p = assemble_pressure_system(
        cell_centers, face_info, k_cells, phi_cells, Sw, mu_w, mu_o, wells, dt
    )
    p = spla.spsolve(A_p, b_p)

    Sw_new = solve_transport(cell_centers, face_info, phi_cells, Sw, p, mu_w, dt)

    Sw = Sw_new

    print(f"Шаг {step + 1}: Насыщенность [{Sw.min():.3f}, {Sw.max():.3f}]")

plot_results(points, triangles, p, Sw)

mesh_out = meshio.Mesh(
    points, [("triangle", triangles)], cell_data={"Pressure": [p], "Saturation": [Sw]}
)
mesh_out.write("results.vtk")
