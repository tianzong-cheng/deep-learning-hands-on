import numpy as np
import torch
import torch.nn as nn

# Generate sine wave data
seq_length = 20
num_samples = 1000

time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)

# Remove the last element, making the array exactly a period
# Then reshape the array into a 2D-array
data = data[:-1].reshape(-1, seq_length)

# Repeat the period for 1000 times
X = np.repeat(data, num_samples, axis=0)
y = np.sin(time_steps[-1] * np.ones((num_samples, 1)))  # Every element of y is sin(pi)

# Convert to PyTorch tensors
# Adds a new dimension at index 2
# Final size: 1000, 20, 1
# Default LSTM layer input: sequence length, batch size, input size
X_tensor = torch.FloatTensor(X).unsqueeze(2)
y_tensor = torch.FloatTensor(y)

class TimeSeries_LSTM(nn.Module):
    def __init__(self):
        super(TimeSeries_LSTM, self).__init__()
        self.lstm = nn.LSTM(1, 50)  # 1 input feature, 50 hidden units
        self.fc = nn.Linear(50, 1)  # 50 input features, 1 output feature

    def forward(self, x):
        # Initial Hidden State
        h_0 = torch.zeros(1, x.size(1), 50)
        # Initial Cell State
        c_0 = torch.zeros(1, x.size(1), 50)
        out, _ = self.lstm(x, (h_0, c_0))  # We don't care about the intermediate states, so they are discarded

        # Take the last time step of the LSTM output and pass it through a fully connected layer
        # This is common approach when using LSTMs for sequence-to-one tasks
        out = self.fc(out[:, -1, :])
        return out

# Initialize
model = TimeSeries_LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.6f}')

# Generate data for testing

num_test_samples = 100

test_time_steps = np.linspace(0, np.pi, seq_length + 1)
test_data = np.sin(test_time_steps)
test_data = test_data[:-1].reshape(-1, seq_length)

test_X = np.repeat(test_data, num_test_samples, axis=0)
test_y = np.sin(test_time_steps[-1] * np.ones((num_test_samples, 1)))

test_X_tensor = torch.FloatTensor(test_X).unsqueeze(2)
test_y_tensor = torch.FloatTensor(test_y)


with torch.no_grad():
    predicted = model(test_X_tensor)

    mse = criterion(predicted, test_y_tensor).item()
    print(f'Mean Squared Error on test data: {mse:.6f}')

predicted_np = predicted.numpy()

# Visualize the result

import matplotlib.pyplot as plt

plt.plot(test_time_steps[:-1], test_X[0], label='Input Sequence')
plt.scatter([test_time_steps[-1]], [test_y[0]], label='Actual Future Value', c='r')
plt.scatter([test_time_steps[-1]], [predicted_np[0]], label='Predicted Future Value', c='b')
plt.legend()
plt.show()

