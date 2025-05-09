import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic data: y = x + 1
torch.manual_seed(0)
X = torch.randn(1000, 1, 20, 20)
y = X + 1.0

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# Training loop
model.train()
for epoch in range(20):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

#Save model and test data
# torch.save(model.state_dict(), "saved_cnn-simple_cnn.pt")
# torch.save({"X_test": X[:3], "y_test": y[:3]}, "test_samples.pt")
# print("Model and test samples saved.")

# If you know you are going to run on CPU, just script the model as so.
model.eval()
scripted_model = torch.jit.script(model)
model_path = "saved_cnn-simple_model_cpu.pt"
scripted_model.save(model_path)
