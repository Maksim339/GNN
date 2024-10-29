import numpy as np
import solver
import os
from itertools import product

N = 10
output_directory = "pressure_dataset"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def generate_pressure_dataset(N, output_directory):
    sample_idx = 0
    for injection_pos in product(range(1, N - 1), repeat=2):
        for production_pos in product(range(1, N - 1), repeat=2):
            if injection_pos != production_pos:
                pressure_field = solver.solve_pressure_field(N, 1.0, 0.0, 1.0, injection_pos, production_pos)
                print(pressure_field)
                output_file = os.path.join(output_directory, f"pressure_sample_{sample_idx}.npy")
                np.save(output_file, pressure_field)

                well_positions = {
                    'injection': injection_pos,
                    'production': production_pos
                }
                well_positions_file = os.path.join(output_directory, f"well_positions_{sample_idx}.npy")
                np.save(well_positions_file, well_positions)

                print(f"Generated sample {sample_idx + 1}")
                sample_idx += 1


generate_pressure_dataset(N, output_directory)
