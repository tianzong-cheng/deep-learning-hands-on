import torch
import torch.nn as nn
import torch.optim as optim
import numpy as numpy

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

x_test = torch.tensor([[0.1, 0.0], [0.0, 0.9], [1.1, 0.0], [0.9, 1.0]], dtype=torch.float32)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 2 input features, 2 output features
        self.layer1 = nn.Linear(2, 2)
        # 2 input features, 1 output feature
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        # Use sigmoid activation function
        # It can be interpreted as a continous version of threshold function
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

# Initialize an instance of the model
model = SimpleNN()
# Use Mean Squared Error Loss, which is suitable for regression problems
criterion = nn.MSELoss()
# Use Stochastic Gradient Descent with learning rate 0.1 to minimize the loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10000):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print Loss
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

with torch.no_grad():
    test_outputs = model(x_test)
    print(f'Test Outputs: {test_outputs}')

