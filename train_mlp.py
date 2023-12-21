import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from helpers import *

path = sys.argv[1] + "/"
model_name = sys.argv[2]

S = np.load(path + "S.npy")
F = np.load(path + "F.npy")
H = np.load(path + "H.npy")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert them to PyTorch tensors
S = torch.from_numpy(S).float().to(device)
F = torch.from_numpy(F).float().to(device)
H = torch.from_numpy(H).float().to(device)


# Separate into training and testing set
split = 0.9
num_entries = len(S)
train_len = int(num_entries * split)

S_train = S[:train_len]
F_train = F[:train_len]

S_test = S[train_len:]
F_test = F[train_len:]

batch_size = 1024

train_dataset = MatrixDataset(S_train, F_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MatrixDataset(S_test, F_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

N_test, _ = np.shape(S_test)
N, n_b = np.shape(S_train)
_, n = np.shape(F_train)

input_size = n_b
output_size = n

# Define network
net = MatrixFactorizationNet(input_size, [512, 256, 128], output_size)

net.to(device)

# Define Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_losses = []
val_losses = []
epochs = []

num_epochs = 1000  # Define the number of epochs
for epoch in range(num_epochs):
    net.train()
    total_loss = 0.0
    for S_batch, F_batch in train_loader:
        optimizer.zero_grad()

        S_batch_flattened = S_batch.to_dense().view(S_batch.size(0), -1)

        # Forward pass
        F_batch_pred = net(S_batch_flattened)
        F_batch_pred_normalized = F_batch_pred / (F_batch_pred.sum(dim=1, keepdim=True) + 1e-8) # To avoid divisions by zero

        # Compute approximation of S and ground truth S
        SH_batch = torch.matmul(F_batch_pred_normalized, H)
        S_true = torch.matmul(F_batch, H)

        # Compute loss
        S_loss = criterion(SH_batch, S_true)
        F_loss = criterion(F_batch_pred_normalized, F_batch)
        loss = 5000 * F_loss + S_loss #We multiply by 5000 because F loss is very small compared to S loss
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        total_val_loss = 0.0

        for S_batch, F_batch in test_loader:
            S_batch_flattened = S_batch.to_dense().view(S_batch.size(0), -1)

            F_batch_pred = net(S_batch_flattened)
            F_batch_pred = apply_sparsity(F_batch_pred)
            F_batch_pred_normalized = F_batch_pred / F_batch_pred.sum(dim=1, keepdim=True)

            # Compute approximation of S and ground truth S
            SH_batch = torch.matmul(F_batch_pred_normalized, H)
            S_true = torch.matmul(F_batch, H)

            S_loss = criterion(SH_batch, S_true)
            F_loss = criterion(F_batch_pred_normalized, F_batch)
            loss = 5000 * F_loss + S_loss
            total_val_loss += loss.item()

        # Calculate the average loss over all test samples
        avg_loss = total_val_loss / len(test_loader)

    # Print the average loss for this epoch
    average_loss = total_loss / len(train_loader)
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_loss}')
        train_losses.append(average_loss)
        val_losses.append(avg_loss)
        epochs.append(epoch + 1)

# Save the model
torch.save(net.state_dict(), "trained_model/" + path + model_name)
