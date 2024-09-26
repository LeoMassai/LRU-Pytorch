import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from LRU import LRU, DeepLRU


# Function to generate time series data with a sine wave, trend, and noise
def generate_time_series(batch_size, seq_length=40, input_size=3):
    X = torch.zeros(batch_size, seq_length, input_size)
    Y = torch.zeros(batch_size, seq_length, 1)  # We are predicting one future value

    for i in range(batch_size):
        # Create a time series for each sample in the batch
        time_steps = np.linspace(0, 4 * np.pi, seq_length)  # Create a range of time steps

        # Generate a sine wave with a trend and some noise
        sine_wave = np.sin(time_steps)  # Sine wave
        trend = time_steps * 0.1  # Linear trend (gradual increase)
        noise = np.random.normal(0, 0.1, seq_length)  # Random noise

        # Combine to form the time series
        series = sine_wave + trend + noise

        # Populate the input sequence (previous steps as the model input)
        for j in range(input_size):
            X[i, :, j] = torch.tensor(series)

        # Target is the next value in the series (shifted by one time step)
        Y[i, :-1, 0] = torch.tensor(series[1:])
        Y[i, -1, 0] = torch.tensor(series[-1])  # For simplicity, the last target is the last input

    return X, Y


# Example model: A simple RNN for time series prediction
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeSeriesModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        rnn_out, _ = self.rnn(x)  # Output from the RNN layer
        output = self.fc(rnn_out)  # Apply the fully connected layer to the RNN output
        return output


# Instantiate the model
input_size = 3
hidden_size = 20
output_size = 1
#model = TimeSeriesModel(input_size, hidden_size, output_size)


# Instantiate the model
N = 4
model = DeepLRU(N,
                in_features=input_size,
                out_features=output_size,
                state_features=hidden_size
                )

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, num_epochs=100, batch_size=16, seq_length=40, input_size=3):
    model.train()

    for epoch in range(num_epochs):
        # Generate time series data for this epoch
        inputs, targets = generate_time_series(batch_size, seq_length, input_size)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Example training
train_model(model)


# Generate and visualize an example time series and prediction
def visualize_prediction(model, seq_length=40, input_size=3):
    model.eval()
    with torch.no_grad():
        # Generate a single time series
        inputs, targets = generate_time_series(1, seq_length, input_size)
        predictions = model(inputs).squeeze().cpu().numpy()
        targets = targets.squeeze().cpu().numpy()
        inputs = inputs.squeeze().cpu().numpy()

        # Plot the input time series, target, and prediction
        plt.figure(figsize=(10, 6))
        plt.plot(range(seq_length), inputs[:, 0], label="Input Time Series", color='blue')
        plt.plot(range(seq_length), targets, label="True Future", color='green')
        plt.plot(range(seq_length), predictions, label="Predicted Future", color='red')
        plt.legend()
        plt.title("Time Series Prediction")
        plt.show()


# Visualize the model prediction on one example time series
visualize_prediction(model)

a= model(torch.randn([4, 3]))


