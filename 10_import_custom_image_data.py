# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms
from customDatasetLoader import ImageClassifierDatasetLoader

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
in_channels = 3
n_classes = 10
learning_rate = 1e-3
batch_size = 32
n_epochs = 1


# data downloaded from https://www.kaggle.com/c/dogs-vs-cats/data
# see the datasets/kaggle_cats_dogs for a script to generate the csv files used to classify this data
# and another to resize images to be of the same size as required for our model
dataset = ImageClassifierDatasetLoader(
    csv_file="datasets/kaggle_cats_dogs/train.csv",
    root_dir="datasets/kaggle_cats_dogs/train_resized",
    transform=transforms.ToTensor(),
)
print("Length of the dataset: ", dataset.__len__())

# loading data; we're sub-setting a validation set from the training set itself
train_set, test_set = torch.utils.data.random_split(dataset, [20000, 4999])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# model; using a preloaded torchvision model
model = models.googlenet(weights="DEFAULT")
model.to(device)

# loss and optimiser
loss_calculator = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(n_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = loss_calculator(scores, targets)

        # backward
        optimiser.zero_grad()
        loss.backward()

        # optimiser
        optimiser.step()


def check_accuracy(loader, model):
    n_correct = 0
    n_samples = 0
    model.eval()  # move from training mode to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)  # [64, 10]
            _, predictions = scores.max(1)
            n_correct += (predictions == y).sum()
            n_samples += predictions.size(0)

        print(
            f"{n_correct} out of {n_samples} are correct; accuracy = {float(n_correct)/float(n_samples)*100:.2f} %."
        )


# these printed statements aren't in the check_accuracy function this time as our train_loader doesn't have any
# attribute defined to decide between a testing and training set
print("Checking accuracy on training data:")
check_accuracy(train_loader, model)
print("Checking accuracy on testing data.")
check_accuracy(test_loader, model)
