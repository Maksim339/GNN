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
        k_harm = 3 / (
            1 / k_vertex[tri[0]] + 1 / k_vertex[tri[1]] + 1 / k_vertex[tri[2]] + 1e-12
        )
        k_cells[i] = k_harm
        phi_cells[i] = (
            phi_vertex[tri[0]] + phi_vertex[tri[1]] + phi_vertex[tri[2]]
        ) / 3
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
            L_edge = np.linalg.norm(edge_vec)
            if L_edge == 0:
                continue
            normal = np.array([edge_vec[1], -edge_vec[0]])
            norm_length = np.linalg.norm(normal)
            if norm_length != 0:
                normal /= norm_length
            center_i = cell_centers[i]
            face_center = 0.5 * (p1 + p2)
            if np.dot(normal, face_center - center_i) > 0:
                normal = -normal

            face_info[i].append((j, normal, L_edge))
            face_info[j].append((i, -normal, L_edge))
    return face_info


def harmonic_avg(a, b, eps=1e-12):
    return 2 * a * b / (a + b + eps)


def assemble_pressure_system(
    cell_centers, face_info, k_cells, Sw, mu_w, mu_o, wells, p_inj, p_prod
):
    M = len(cell_centers)
    A = sp.lil_matrix((M, M))
    b = np.zeros(M)
    eps = 1e-12
    krw, kro = rel_perm(Sw)
    lambda_t = krw / mu_w + kro / mu_o

    for i in range(M):
        for j, normal, L_edge in face_info[i]:
            if j <= i:
                continue
            d = np.linalg.norm(cell_centers[j] - cell_centers[i])
            if d < eps:
                continue
            k_avg = harmonic_avg(k_cells[i], k_cells[j], eps)
            lambda_avg = harmonic_avg(lambda_t[i], lambda_t[j], eps)
            T_ij = (L_edge / d) * k_avg * lambda_avg
            A[i, i] += T_ij
            A[j, j] += T_ij
            A[i, j] -= T_ij
            A[j, i] -= T_ij
    inj_cell = wells["inj"]["cell"]
    A[inj_cell, :] = 0
    A[inj_cell, inj_cell] = 1.0
    b[inj_cell] = p_inj
    prod_cell = wells["prod"]["cell"]
    A[prod_cell, :] = 0
    A[prod_cell, prod_cell] = 1.0
    b[prod_cell] = p_prod
    return A.tocsr(), b


def solve_transport(cell_centers, face_info, k_cells, phi, Sw, p, mu_w, wells, dt):
    M = len(cell_centers)
    net_flux = np.zeros(M)
    eps = 1e-12
    krw, _ = rel_perm(Sw)
    lambda_w = krw / mu_w
    inj_cell = wells["inj"]["cell"]
    lambda_w[inj_cell] = 1.0 / mu_w
    for i in range(M):
        for j, normal, L_edge in face_info[i]:
            if j <= i:
                continue
            d = np.linalg.norm(cell_centers[j] - cell_centers[i])
            if d < eps:
                continue
            k_avg = harmonic_avg(k_cells[i], k_cells[j], eps)
            dp = p[i] - p[j]
            if dp > 0:
                lam_up = lambda_w[i]
            else:
                lam_up = lambda_w[j]
            T_w = (L_edge / d) * k_avg * lam_up
            F = T_w * dp
            net_flux[i] -= F
            net_flux[j] += F
    Sw_new = Sw + dt * (net_flux / phi)
    Sw_new[inj_cell] = 1.0
    Sw_new = np.clip(Sw_new, 0.0, 1.0)
    return Sw_new


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


# -----------------------------------------------------------------------
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

mu_w = 0.001
mu_o = 1.0
dt = 0.01
total_time = 5.0
num_steps = int(total_time / dt)

wells = {"inj": {"cell": 233}, "prod": {"cell": 114}}
p_inj = 10.0
p_prod = 0.0

cell_centers, k_cells, phi_cells = build_cell_data(
    points, triangles, mesh_perm.point_data["PERMEABILITY"], mesh_phi.point_data["PHI"]
)
face_info = build_connectivity(points, cell_centers, triangles)

Sw = np.full(len(cell_centers), 0.2)
Sw[wells["inj"]["cell"]] = 1.0
p = np.zeros(len(cell_centers))

for step in range(num_steps):
    A_p, b_p = assemble_pressure_system(
        cell_centers,
        face_info,
        k_cells,
        Sw,
        mu_w,
        mu_o,
        wells,
        p_inj,
        p_prod,
    )
    p = spla.spsolve(A_p, b_p)
    Sw_new = solve_transport(
        cell_centers, face_info, k_cells, phi_cells, Sw, p, mu_w, wells, dt
    )
    Sw = Sw_new
    print(f"Шаг {step+1}: S_w min = {Sw.min():.3f}, max = {Sw.max():.3f}")

plot_results(points, triangles, p, Sw)

mesh_out = meshio.Mesh(
    points, [("triangle", triangles)], cell_data={"Pressure": [p], "Saturation": [Sw]}
)
mesh_out.write("results1.vtk")
