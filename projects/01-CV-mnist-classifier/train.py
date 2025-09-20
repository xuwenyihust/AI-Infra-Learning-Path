import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Hyperparameters & Configuration
BATCH_SIZE = 64
EPOCHS = 5  # Rounds of training
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "mnist_cnn.pth"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Data Preparation
print("Preparing dataset...")

# Define the data transformation pipeline.
transform = transforms.Compose([
    # Converts a PIL Image or numpy.ndarray to a Tensor,
    # and scales the pixel values from [0, 255] to [0.0, 1.0].
    transforms.ToTensor(),
    # Normalizes a Tensor with a given mean and standard deviation.
    # This helps the model converge faster.
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training dataset.
# Applies the defined transformation pipeline
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader.
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Dataset prepared.")

# Model Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # PyTorch expects image tensors in the format (Batch, Channels, Height, Width).
        # MNIST images are 1x28x28 (1 grayscale channel).

        # First convolutional layer
        # in_channels=1: Input is a grayscale image.
        # out_channels=32: Produces 32 feature maps (using 32 kernels).
        # kernel_size=3: Uses a 3x3 convolutional kernel.
        # padding=1: Preserves the image dimensions (28x28) after convolution.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # Second convolutional layer
        # in_channels=32: Must match the out_channels of the previous layer.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Max pooling layer
        # For downsampling feature maps.
        # This layer slides a window (kernel) over the feature maps.
        # Keep the max value within the window and discard the rest.
        # kernel_size=2: Uses a 2x2 window.
        # stride=2: Halves the feature map dimensions (28x28 -> 14x14 -> 7x7).
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected (Linear) layer
        # in_features=64*7*7: Flattens the output from the pooling layer. 64 channels * 7x7 size.
        # out_features=128: Outputs 128 neurons.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # Output layer
        # in_features=128: Matches the output of the previous layer.
        # out_features=10: Outputs 10 classes for digits 0-9.
        self.fc2 = nn.Linear(128, 10)

        # Activation function
        self.relu = nn.ReLU()

    # Define the forward pass of the model.
    def forward(self, x):
        # Convolution -> Activation -> Pooling
        x = self.pool(self.relu(self.conv1(x)))
        # Convolution -> Activation -> Pooling
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten the feature map for the fully connected layer.
        # x.view() is similar to numpy's reshape.
        # -1 infers the dimension size from the other dimensions (here, it's the batch_size).
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layer -> Activation
        x = self.relu(self.fc1(x))

        # Output layer (Note: No Softmax here, as nn.CrossEntropyLoss applies it internally).
        x = self.fc2(x)
        return x

# Training Process
def train_model():
    print("Initializing model...")
    model = SimpleCNN().to(DEVICE)

    # Define the loss function: Cross-Entropy Loss is suitable for multi-class classification.
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer: Adam is a popular and effective optimization algorithm.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    # The main training loop
    for epoch in range(EPOCHS):
        # Set the model to training mode.
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            # 1. Move data and labels to the selected device (GPU or CPU).
            data, target = data.to(DEVICE), target.to(DEVICE)

            # 2. Zero the gradients to prevent accumulation from previous batches.
            optimizer.zero_grad()

            # 3. Forward pass: compute predicted outputs by passing inputs to the model.
            output = model(data)

            # 4. Calculate the batch loss.
            loss = criterion(output, target)

            # 5. Backward pass: compute gradient of the loss with respect to model parameters.
            loss.backward()

            # 6. Perform a single optimization step (parameter update).
            optimizer.step()

            # Print training statistics.
            if batch_idx % 200 == 0:
                print(
                    f"Epoch: {epoch + 1}/{EPOCHS} | Batch: {batch_idx:4d}/{len(train_loader)} | Loss: {loss.item():.6f}")

    print("Training finished.")

    # Save the model.
    # We only save the model's 'state_dict', which contains all weights and biases.
    # This is the recommended and most flexible way to save a model.
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


# --- 5. Main Execution Block ---
# This ensures the training code runs only when the script is executed directly.
if __name__ == '__main__':
    train_model()




