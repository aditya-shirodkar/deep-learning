# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# simple fully-connected NN
class NN(nn.Module):
    def __init__(self, input_size, n_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# simple convolutional NN
# Convolution is using a kernel to extract features from an image.
# The kernel is a matrix which slides across an image and multiplies (dot product) with the input in order to enhance
# certain features of the output. The kernel moves across the image with its stride. As the kernel would normally
# reduce the size of the output, padding may be added to the image to offset this.
# Two types of padding: valid padding (i.e. no padding, the output image would shrink) and same padding
# (i.e. adding 0-value pixels around the currently scanned pixel, enough to preserve the size of the image).
# A same convolution is when the number of input dimensions equals the number of output dimensions, achieved using
# same padding, for instance.
class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()
        # For n_in = input dimensions, n_out = output dimensions, k = kernel size (1 dimension), p = padding, s = stride
        # n_out = floor((n_in + 2p - k)/s) + 1
        # For the MNIST dataset and the values set below, n_in = 28, k = 3, p = 1, s = 1; therefore n_out = 28
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=8,  # arbitrarily chosen here, increasing channels for more information
            kernel_size=(
                3,
                3,
            ),  # standard 3x3 kernel moving across the pixels of the image
            stride=(
                1,
                1,
            ),  # steps taken by the kernel on either dimension (low stride increases computation)
            padding=(1, 1),  # same padding
        )
        # pooling slides a filter (similar to kernel) OVER EACH CHANNEL and summarises them into a single value
        # in this case, a maxpool is used, which picks the maximum value in the filter.
        # the filter below covers 2x2 squares at a time, with no overlap (due to the stride). There's no padding in
        # pooling, so, n_out = floor((n_in - k)/s) + 1. Here, n_in = 28, k = 2, s = 2; therefore n_out = 14
        # i.e. it halves the dimensions.
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # the forward function uses pooling twice, so we can expect both "image" (non-channel) dimensions to get
        # halved twice from 28 to 7. The linear function expects the flattened number of dimensions, therefore:
        self.fc1 = nn.Linear(16 * 7 * 7, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(
            x.shape[0], -1
        )  # keep mini-batch (number of samples sent in), flatten rest
        return self.fc1(x)


# testing
model = CNN(1, 10)
# batch_size = 64, n_channels = 1, image dims = (28, 28); unlike the simple NN, we aren't flattening the non-batch dims
data = torch.randn(64, 1, 28, 28)
print(model(data).shape)  # expected (64, 10)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
in_channels = 1
n_classes = 10
learning_rate = 1e-3
batch_size = 64
n_epochs = 1

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

# train network
for epoch in range(n_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        # no need to flatten data here

        # forward
        scores = model(data)
        loss = loss_calculator(scores, targets)

        # backward
        optimiser.zero_grad()
        loss.backward()

        # optimiser
        optimiser.step()


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
