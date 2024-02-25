# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
# to save a model and continue training at another time; saving at specific epochs here
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
in_channels = 1
n_classes = 10
learning_rate = 1e-3
batch_size = 64
n_epochs = 5
load_model = True
filename = "savefiles/tutorial_07_checkpoint.pth.tar"  # to save current model state


# CNN
class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc1(x)


# function to store checkpoint
def save_checkpoint(state, filename=filename):
    print("Saving checkpoint...")
    torch.save(state, filename)
    print("Checkpoint saved at ", filename)


def load_checkpoint(checkpoint):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser"])
    print("Loaded checkpoint.")


# load data
x_train = datasets.MNIST(
    root="datasets/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
x_test = datasets.MNIST(
    root="datasets/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=True)

# initialise CNN
model = CNN(in_channels, n_classes).to(device)

# loss and optimiser
loss_calculator = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# loads model; if you're looking to restart training, be sure to delete the old checkpoint or change the filename
# or else each training loop will add more epochs to the already trained model!
if load_model and os.path.exists(filename):
    load_checkpoint(torch.load(filename))

# train network
for epoch in range(n_epochs):
    losses = []
    if epoch % 2 == 0:
        # you could add a lot more info if you'd like to your checkpoint, but be sure to load that info too
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimiser": optimiser.state_dict(),
        }
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = loss_calculator(scores, targets)
        losses.append(loss.item())

        # backward
        optimiser.zero_grad()
        loss.backward()

        # optimiser
        optimiser.step()

    print(f"Loss after epoch {epoch}: {np.sum(losses)*100/(batch_idx+1):.2f} %.")


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data:")
    else:
        print("Checking accuracy on testing data.")
    n_correct = 0
    n_samples = 0
    model.eval()  # move from training mode to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # no reshape in this accuracy check

            scores = model(x)  # [64, 10]
            _, predictions = scores.max(
                1
            )  # index of max value in second (class) dimension
            n_correct += (predictions == y).sum()
            n_samples += predictions.size(0)

        print(
            f"{n_correct} out of {n_samples} are correct; accuracy = {float(n_correct)/float(n_samples)*100:.2f} %."
        )

    model.train()  # return to training mode


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
