import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- 1. LOAD AND PREPARE DATA ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.load('tandem_train_data.pt')

# Normalize X: Thicknesses (0 to 400nm) -> (0 to 1)
X = data['X'] / 400.0 
Y = data['Y'] # Transmittance (already 0 to 1)

# Split into Train (90%) and Validation (10%)
dataset = TensorDataset(X, Y)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# --- 2. ARCHITECTURE: THE NEURAL SIMULATOR ---
class ForwardNet(nn.Module):
    def __init__(self):
        super(ForwardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(23, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            
            nn.Linear(1024, 2048), # Increased width for spectral resolution
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            
            nn.Linear(1024, 1000),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

model = ForwardNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.1)

# --- 3. TRAINING LOOP ---
epochs = 500
print(f"Training Forward Simulator on {device}...")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            val_loss += criterion(model(bx), by).item()
    
    avg_val = val_loss/len(val_loader)
    scheduler.step(avg_val)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {avg_val:.6f}")

# --- 4. SAVE THE "PHYSICS BRAIN" ---
torch.save(model.state_dict(), 'forward_simulator.pth')
print("\nForward Simulator trained and saved as 'forward_simulator.pth'")