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


# create a bidirectional LSTM RNN
# In a bidirectional LSTM, the layers are traversed both ways (e.g. a sentence being read from both sides) to achieve
# more information such as capturing more abstract dependencies. For more dimensions, this behaviour can be further
# compounded.
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_classes):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers, batch_first=True, bidirectional=True
        )
        # as the operation is two-ways across layers, the hidden size is updated once either way, and therefore we
        # must expect twice the features in the fully-connected circuit
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # the operation is two-ways across layers, we need to double the layers dimension for the hidden and cell states
        h0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
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
model = BidirectionalLSTM(input_size, hidden_size, n_layers, n_classes).to(device)

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
