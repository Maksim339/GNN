import numpy as np
import matplotlib.pyplot as plt
import os

dataset_directory = "pressure_dataset"


def visualize_pressure_field(pressure_file_name):
    pressure_file_path = os.path.join(dataset_directory, pressure_file_name)
    sample_idx = pressure_file_name.split('_')[-1].split('.')[0]
    well_positions_file_path = os.path.join(dataset_directory, f"well_positions_{sample_idx}.npy")

    pressure_field = np.load(pressure_file_path)
    print(pressure_field)

    well_positions = np.load(well_positions_file_path, allow_pickle=True).item()
    injection_pos = well_positions['injection']
    production_pos = well_positions['production']

    plt.imshow(pressure_field, origin='lower', extent=[0, pressure_field.shape[0], 0, pressure_field.shape[1]],
               cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.title(f'Pressure Field: {pressure_file_name}')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter([injection_pos[0]], [injection_pos[1]], color='red', label='Injection Well')
    plt.scatter([production_pos[0]], [production_pos[1]], color='blue', label='Production Well')

    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    pressure_file_name = input("enter the pressure filename ('pressure_sample_0.npy'): ")

    pressure_file_path = os.path.join(dataset_directory, pressure_file_name)
    if not os.path.exists(pressure_file_path):
        print(f"'{pressure_file_name}' not found.")
        return
    visualize_pressure_field(pressure_file_name)


if __name__ == "__main__":
    main()
