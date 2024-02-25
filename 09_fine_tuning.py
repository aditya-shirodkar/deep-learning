# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
in_channels = 3
n_classes = 10
learning_rate = 1e-3
batch_size = 1024
n_epochs = 5

# we here attempt to improve the accuracy and/or speed of the previous model in the previous file
model = models.vgg16(weights="VGG16_Weights.DEFAULT")


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(selfs, x):
        return x


# Here, we're removing backpropagation for every stage, so that our new training doesn't affect it. Instead, we want to
# affect only the final few layers while leaving the rest of the model's weights frozen. Generally, earlier layers are
# the ones which capture more abstract/lower level features which are less understandable by us, and therefore may be
# difficult to effectively modify.
for param in model.parameters():
    param.requires_grad = False

# As these stages are added after the above step, they still will conduct backpropagation. Adding another linear layer
# and a ReLU layer, and no longer removing the final maxpool feature:
model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 10))
model.to(device)
# we can expect running this model to be faster than the previous, but not as accurate due to the backpropagation
# freeze implemented for the earlier layers.

# load data
x_train = datasets.CIFAR10(
    root="datasets/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
x_test = datasets.CIFAR10(
    root="datasets/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=True)

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
