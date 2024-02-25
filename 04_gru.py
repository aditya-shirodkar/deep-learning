# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 28
sequence_length = 28
n_layers = 2
hidden_size = 256
n_classes = 10
learning_rate = 1e-3
batch_size = 64
n_epochs = 1


# create a gated recurrent unit RNN
# A GRU uses gating mechanisms to selectively update the hidden state of the RNN at each time step. This addresses the
# "vanishing gradient" problem with RNNs. Basically, the lower the gradient (from your optimiser) is, the more limited
# is the learning capacity of the RNN, the harder it is to capture long-term dependencies, convergence is slower, and
# there's a chance of preferential learning in shallow networks. Also look at the "exploding gradient" problem.
# A GRU has two gating mechanisms:
# 1) the reset gate: determines how much of the previous hidden state should be forgotten
# 2) the update gate: determines how much of the new input should be used to update the hidden state
# GRUs were introduced as a simpler alternative to using long short term memory (LSTM).
# In PyTorch, the modification is simple: swap out the nn.RNN function for the nn.GRU one, using the same parameters.
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * sequence_length, n_classes)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)


# load data
x_train = datasets.MNIST(
    root="datasets/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
x_test = datasets.MNIST(
    root="datasets/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=True)

# initialise GRU
model = GRU(input_size, hidden_size, n_layers, n_classes).to(device)

# loss and optimiser
loss_calculator = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(n_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = loss_calculator(scores, targets)

        # backward
        optimiser.zero_grad()
        loss.backward()
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
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            n_correct += (predictions == y).sum()
            n_samples += predictions.size(0)

        print(
            f"{n_correct} out of {n_samples} are correct; accuracy = {float(n_correct)/float(n_samples)*100:.2f} %."
        )

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
