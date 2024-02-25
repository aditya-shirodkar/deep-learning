# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST dataset shape is batch_size x n_channels x number of time sequences x n_features => 64 x 1 x 28 x 28
# hyperparameters
input_size = 28
sequence_length = 28
n_layers = 2  # number of RNN layers
hidden_size = 256  # nodes in hidden layer
n_classes = 10
learning_rate = 1e-3
batch_size = 64
n_epochs = 1


# create a basic recurrent neural network
# RNNs aren't generally used for images, but we continue with MNIST here for testing.
# RNNs use sequential (time series) data, where the output from the previous step is fed as input to the current step.
# RNNs account for the positions of elements to make sense of the data, for instance, a sentence must be expressed
# in the order of its words for it to mean what is intended.
# An important element is the "hidden state" (or memory state) which remembers information about a particular sequence.
# Unlike feed-forward networks, RNNs share the same weight parameter within every layer of the network
# (i.e. in the hidden state; they still are adjusted via backpropagation and gradient descent). The hidden state
# keeps improving as the model trains, and being reused across every layer lowers the comparative complexity of an RNN.
# RNN networks generally loop into themselves, whereas in feed-forward networks, all information is perpetually passed
# forwards. Therefore, feed-forward networks are preferred for image classification tasks rather than sequential data
# analysis tasks, where RNNs are preferred.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(
            input_size,  # features processed per time step; runs till completion, so don't need to specify time steps
            hidden_size,
            n_layers,
            batch_first=True,  # as our first axis is the batch dimension
        )
        # fully-connected linear layer uses info from every hidden state for every time sequence
        self.fc = nn.Linear(hidden_size * sequence_length, n_classes)

    def forward(self, x):
        # initialise hidden state; n_layers x mini-batch size x hidden_size
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)

        # every sample has its own hidden state, so we ignore that output with _
        out, _ = self.rnn(x, h0)
        # keep batch, flatten rest
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)


# testing -> here, unlike the previous, as our RNN has an internal element on our device
# we must set  the model and data to device too. generally good practice anyway
model = RNN(input_size, hidden_size, n_layers, n_classes).to(device)
# data expected (see h0 in forward) is batch_size x sequence_length x input_size
data = torch.randn(64, 28, 28).to(device)
print(model(data).shape)  # expected (64, 10)

# load data
x_train = datasets.MNIST(
    root="datasets/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
x_test = datasets.MNIST(
    root="datasets/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=True)

# initialise RNN
model = RNN(input_size, hidden_size, n_layers, n_classes).to(device)

# loss and optimiser
loss_calculator = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(n_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # RNN expects data of size batch_size x 28 x 28 (see forward function), so we squeeze out the channel dimension
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = loss_calculator(scores, targets)

        # backward
        optimiser.zero_grad()
        loss.backward()
        # always clip the gradients for all RNN based neural networks, to prevent the problem of "exploding gradients."
        # Gradients which have exploded have very high values, causing them to unfairly affect the network. It's good
        # to "clip" them (or rather, their norms) by lowering their values if they pass a certain threshold.
        # https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

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
            x = x.to(device).squeeze(1)
            y = y.to(device)

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
