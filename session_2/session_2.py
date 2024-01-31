import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Data transformation, (converting to Tensor and normalizing)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training dataset
train_set = torchvision.datasets.MNIST(root='.data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

# Download and load the test dataset
test_set = torchvision.datasets.MNIST(root='.data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

class MNISTNN(nn.Module):
    def __init__(self):
        super(MNISTNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Reshape the tensor
        x = x.view(-1, 28*28)
        
        # Rectified Linear Unit (ReLU) activation function
        # The output is the maximum of 0 and the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # The output layer does not apply any activation function since we'll use it with Cross-Entropy Loss later
        x = self.fc3(x)
        return x

model = MNISTNN()
criterion = nn.CrossEntropyLoss()  # Commonly used when the output is a probability between 0 and 1
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(5):
    for i, data in enumerate(train_loader, 0):  # Enumerate, starting from 0
        inputs, labels = data

        # PyTorch, by default, accumulates gradients on subsequent backward passes.
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        if i % 500 == 0:
            print(f'Epoch [{epoch + 1}/5], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)

        # Find the maximum values along the second dimension
        # The result are the values, which are ignored, and the indices
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        # sum() computes the sum of all elements in a tensor
        # Since it is applied to a boolean tensor here, it counts the number of True values
        # Then item() converts the result to a Python integer
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
