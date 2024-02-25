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


# create a long short term memory RNN
# Like a GRU, an LSTM captures long-term dependencies and solves other issues related to the vanishing gradient problem.
# The architecture is more complicated than that of a GRU, as the memory cell is controlled by three gates:
# 1) the input gate: controls what information is added to the memory cell
# 2) the forget gate: controls what information is removed from the memory cell
# 3) the output gate: controls what information is output from the memory cell
# Unlike a GRU, "memory" is of two types: long term memory (cell state), and short term memory (hidden state).
# LSTMs are generally better and capture more information but are also more complicated and slower. As they are pretty
# close in performance, GRUs are usually preferred for simplicity.
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
        )
        # now, our last hidden state has all the information of the previous states, and so we don't need to linearly
        # concatenate every previous hidden state (i.e. no need to multiply by sequence*length)
        # though there is *some* information loss, it's not significant and this improves performance significantly
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # hidden state (short term memory)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        # cell state (long term memory)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        # _ here would refer to (hidden_state, cell_state) for every sample, but we only need the final states
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # we only need the last hidden state now
        return out


# load data
x_train = datasets.MNIST(
    root="datasets/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
x_test = datasets.MNIST(
    root="datasets/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=True)

# initialise LSTM
model = LSTM(input_size, hidden_size, n_layers, n_classes).to(device)

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
