import torch
import matplotlib.pyplot as plt
from gnn_model import PressureGNN, val_loader, device
import numpy as np

model = PressureGNN(input_dim=4, hidden_dim=32, output_dim=1)
model.load_state_dict(torch.load('pressure_gnn_model.pth'))
model.to(device)
model.eval()

predicted_pressures = []
true_pressures = []

with torch.no_grad():
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.boundary_mask).squeeze()
        predicted_pressure = out.cpu().numpy().reshape(-1, 10, 10)
        true_pressure_batch = data.y.cpu().numpy().reshape(-1, 10, 10)
        predicted_pressures.append(predicted_pressure)
        true_pressures.append(true_pressure_batch)

predicted_pressures = np.vstack(predicted_pressures)
true_pressures = np.vstack(true_pressures)
print(len(predicted_pressures))
print(len(true_pressures))


def visualize_pressure(predicted, true, idx):
    vmin = min(predicted.min(), true.min())
    vmax = max(predicted.max(), true.max())
    mse = np.mean((true[idx] - predicted[idx]) ** 2)
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'MSE: {mse:.4f}', fontsize=10)
    plt.subplot(1, 2, 1)
    plt.imshow(true[idx], origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Pressure (simulator)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.imshow(predicted[idx], origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Pressure (gnn)')
    plt.grid(True)
    plt.show()


mse = np.mean((true_pressures - predicted_pressures) ** 2)
print(mse, "mse общая")

visualize_pressure(predicted_pressures, true_pressures, idx=700)
