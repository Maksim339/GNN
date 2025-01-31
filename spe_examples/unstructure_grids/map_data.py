import numpy as np
import meshio
from scipy.interpolate import griddata


def map_scalar_field_to_mesh(mesh_vtk, field_vtk, output_vtk, field_name="PHI"):
    mesh = meshio.read(mesh_vtk)
    points_mesh = mesh.points

    field_data = meshio.read(field_vtk)
    points_field = field_data.points
    field_values = field_data.point_data[field_name]

    if (points_field[:, 0].min() != points_mesh[:, 0].min() or
        points_field[:, 0].max() != points_mesh[:, 0].max() or
        points_field[:, 1].min() != points_mesh[:, 1].min() or
        points_field[:, 1].max() != points_mesh[:, 1].max()):
        field_scale_x = (points_mesh[:, 0].max() - points_mesh[:, 0].min()) / \
                        (points_field[:, 0].max() - points_field[:, 0].min())
        field_scale_y = (points_mesh[:, 1].max() - points_mesh[:, 1].min()) / \
                        (points_field[:, 1].max() - points_field[:, 1].min())
        points_field[:, 0] = (points_field[:, 0] - points_field[:, 0].min()) * field_scale_x + points_mesh[:, 0].min()
        points_field[:, 1] = (points_field[:, 1] - points_field[:, 1].min()) * field_scale_y + points_mesh[:, 1].min()


    interpolated_field = griddata(
        points_field[:, :2],
        field_values,
        points_mesh[:, :2],
        method='linear',
        fill_value=np.min(field_values)
    )

    if np.any(np.isnan(interpolated_field)):
        print("NaN значения в интерполированном поле.")
        interpolated_field = np.nan_to_num(interpolated_field, nan=np.min(field_values))

    mesh.point_data[field_name] = interpolated_field

    meshio.write(output_vtk, mesh)
    print(f"сохранён в {output_vtk}")


map_scalar_field_to_mesh(
    mesh_vtk="adaptive_mesh.vtk",
    field_vtk="perm.vtk",
    output_vtk="mesh_perm.vtk",
    field_name="PERMEABILITY"
)
