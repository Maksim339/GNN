import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split


def create_graph_data(pressure_field, well_config):
    if isinstance(well_config, np.ndarray):
        well_config = well_config.item()
    N = pressure_field.shape[0]

    node_features = []
    boundary_mask = []
    for i in range(N):
        for j in range(N):
            k = 1.0

            if (i, j) == tuple(well_config['injection']):
                well = 1
            elif (i, j) == tuple(well_config['production']):
                well = -1
            else:
                well = 0

            node_features.append([i, j, k, well])
            is_boundary = i == 0 or i == N - 1 or j == 0 or j == N - 1
            boundary_mask.append(is_boundary)
    node_features = torch.tensor(node_features, dtype=torch.float)
    boundary_mask = torch.tensor(boundary_mask, dtype=torch.bool)

    edge_index = []
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            if j < N - 1:
                right_idx = i * N + (j + 1)
                edge_index.append([idx, right_idx])
                edge_index.append([right_idx, idx])
            if i < N - 1:
                bottom_idx = (i + 1) * N + j
                edge_index.append([idx, bottom_idx])
                edge_index.append([bottom_idx, idx])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor(pressure_field.flatten(), dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, y=y)
    data.boundary_mask = boundary_mask
    return data


data_list = []
num_samples = len([name for name in os.listdir("pressure_dataset") if name.startswith("pressure_sample_")])
for idx in range(num_samples):
    pressure_field = np.load(f'pressure_dataset/pressure_sample_{idx}.npy', allow_pickle=True)
    well_config = np.load(f'pressure_dataset/well_positions_{idx}.npy', allow_pickle=True)
    data = create_graph_data(pressure_field, well_config)
    data_list.append(data)

train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)


class PressureGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PressureGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, boundary_mask):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x[boundary_mask] = 0.0
        return x


device = torch.device('cpu')
model = PressureGNN(input_dim=4, hidden_dim=32, output_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.boundary_mask).squeeze()
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.boundary_mask).squeeze()
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def main():
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        train_loss = train()
        val_loss = evaluate(val_loader)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Раннее прекращение обучения")
                break

    model.load_state_dict(torch.load('best_model.pth'))

    torch.save(model.state_dict(), 'pressure_gnn_model.pth')


if __name__ == "__main__":
    main()
