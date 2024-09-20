import torch
import torch.nn as nn
import torch.optim as optim

from LRU import LRU, DeepLRU

# Create a single Linear Recurrent Unit, that takes in inputs of size (batch_size, seq_length, 30) (or (seq_length,
# 30)), with internal state-space variable of size 10, and returns outputs of (batch_size, seq_length,
# 50) (or (seq_length, 50)).

lru = LRU(
    in_features=30,
    out_features=50,
    state_features=10
)

#preds = lru(torch.randn([2, 70, 30]))  # Get predictions

#preds

deep = DeepLRU(4,
               in_features=3,
               out_features=5,
               state_features=10
               )

preds = deep(torch.randn([2, 70, 3]))  # Get predictions


# Dummy data generator for input and output
def generate_dummy_data(batch_size, seq_length=40, input_size=3, output_size=5):
    # Input data: Random tensors of shape (batch_size, seq_length, input_size)
    X = torch.randn(batch_size, seq_length, input_size)
    # Target output: Random tensors of shape (batch_size, seq_length, output_size)
    Y = torch.randn(batch_size, seq_length, output_size)
    return X, Y


# Instantiate the model
N = 4
input_size = 3
hidden_size = 20
output_size = 5
model = DeepLRU(N,
                in_features=input_size,
                out_features=output_size,
                state_features=hidden_size
                )

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression task
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, num_epochs=100, batch_size=16, seq_length=40, input_size=3, output_size=5):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        # Generate dummy data for this epoch
        inputs, targets = generate_dummy_data(batch_size, seq_length, input_size, output_size)

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
train_model(model, num_epochs=100, batch_size=16)
