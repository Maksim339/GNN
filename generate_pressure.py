import numpy as np
import solver
import os
import random
from itertools import product

N = 20
output_directory = "pressure_dataset_spe"
permeability_directory = "spe_examples/PERM"
dataset_size = 5000

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def load_permeability_maps(permeability_directory):
    permeability_files = sorted(
        [os.path.join(permeability_directory, f) for f in os.listdir(permeability_directory) if f.endswith(".vtk")]
    )
    return permeability_files


def generate_pressure_dataset(N, permeability_files, output_directory, dataset_size):
    sample_idx = 0
    all_combinations = list(product(range(1, N - 1), repeat=2))
    random.shuffle(all_combinations)

    while sample_idx < dataset_size:
        injection_pos = random.choice(all_combinations)
        production_pos = random.choice(all_combinations)
        if injection_pos == production_pos:
            continue

        perm_filename = random.choice(permeability_files)
        perm_map = solver.read_permeability_from_vtk(perm_filename)

        q_array = solver.build_rhs_q_array(N, injection_pos, production_pos, 1.0, -1.0)
        pressure_field = solver.solve_pressure_field(k_array=perm_map, q_array=q_array, delta_x=0.01)

        output_file = os.path.join(output_directory, f"pressure_sample_{sample_idx}.npy")
        np.save(output_file, pressure_field)

        metadata = {
            'injection': injection_pos,
            'production': production_pos,
            'permeability_file': perm_filename
        }
        metadata_file = os.path.join(output_directory, f"metadata_{sample_idx}.npy")
        np.save(metadata_file, metadata)

        sample_idx += 1


permeability_files = load_permeability_maps(permeability_directory)

generate_pressure_dataset(N, permeability_files, output_directory, dataset_size)
