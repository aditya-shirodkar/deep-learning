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

# load base pytorch pretrained model and modify it
# Weights are used to allow for model pretraining, read more in the docs:
# https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html
# Weights may also be downloaded from elsewhere, often using checkpoints in the process of doing so.
# This notion of loading a pre-trained model and using it for a related task is called TRANSFER LEARNING.
model = models.vgg16(weights="VGG16_Weights.DEFAULT")

# when you print the model, you can see a list of its properties as a dictionary
print(model)


# this class does nothing in its forward stage, thereby allowing you to use it to negate a function
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(selfs, x):
        return x


# removing the final maxpool feature, the entirety of avgpool, and changing the classifier
model.features[30] = Identity()
model.avgpool = Identity()
model.classifier = nn.Linear(2048, 10)
print(model)
model.to(device)
# after running the model, and similarly tweaking it in order to improve accuracy, we're embarking on model FINE-TUNING

# load data (CIFAR10 dataset)
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
