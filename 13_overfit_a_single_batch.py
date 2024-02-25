# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# 1) Overfit a single batch
# Once your model is ready, don't just start training! Instead, send in a single batch as a test for the model to
# overfit a single batch (train and validate on the same model over and over). This is a good way to debug models. You
# can also use tools like overfit_batch in PyTorch Lightning to overfit on the same percentage of batches without
# reshuffling. A model not overfitting is probably bugged, as it is logical that it should.
class NN(nn.Module):
    def __init__(self, input_size, n_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 784
n_classes = 10
learning_rate = 1e-3
# go with batch_size = 1, only then try out your regular batch size (even a single batch should get overfit)
batch_size = 1
n_epochs = 10000  # large number of epochs so the loss can be seen to converge, proving an overfit

# loading data
x_train = datasets.MNIST(
    root="datasets/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)

x_test = datasets.MNIST(
    root="datasets/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=True)

# initialise nn
model = NN(input_size, n_classes).to(device)

# loss and optimiser
calculate_loss = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# get a single batch:
data, targets = next(iter(train_loader))

# send only this single batch to the training loader for n_epochs. Use a large number of epochs to see if the model
# converges to an overfit
for epoch in range(n_epochs):
    print(f"Epoch {epoch} of {n_epochs}")
    # for batch_idx, (data, targets) in enumerate(train_loader):
    data = data.to(device)
    targets = targets.to(device)
    data = data.reshape(data.shape[0], -1)

    # forward
    scores = model(data)
    loss = calculate_loss(scores, targets)
    print(
        f"Loss: {loss}"
    )  # when the printed loss converges to a very low value, the model has overfit

    # backward
    optimiser.zero_grad()
    loss.backward()

    # optimisation
    optimiser.step()


# the accuracy should be low because of the overfit
# the MNIST dataset has digits 0-9 written in different ways, and so sending a batch with a single sample would have it
# learn only one digit quite well and the accuracy could be expected to be slightly over 10%. Even though it's just a
# single variant of a digit being sent, the MNIST dataset is quite simple, so you could expect the model to understand
# that one digit quite well.
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
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

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
