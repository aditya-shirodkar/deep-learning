# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
import torch
import torch.nn as nn
import torch.optim as optim  # optimisation algos e.g. adam, sgd
import torch.nn.functional as F  # functions which don't have parameters e.g. activation functions like relu, tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class NN(nn.Module):
    def __init__(self, input_size, n_classes):
        super(NN, self).__init__()
        # fc stands for fully-connected layer
        self.fc1 = nn.Linear(input_size, 50)  # [784, 50]
        self.fc2 = nn.Linear(
            50, n_classes
        )  # hidden layer; input features must match output of previous

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# testing the above, using same hyperparameters as for the MNIST dataset ahead
model = NN(784, 10)
data = torch.randn(64, 784)  # 64 items with 784 parameters
print("Input shape of data: ", data.shape)
print(
    "Output shape of model: ", model(data).shape
)  # expected [64, 10] -> the 64 items classified into the desired 10 classes

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 784
n_classes = 10
learning_rate = 1e-3  # for the optimiser
batch_size = 64
n_epochs = 1  # the network has seen every point of data in the time of one epoch

# loading data
# Usually you want to send more complicated, sequential transforms to the transform parameter. You can do this with
# transforms.Compose(). For instance, it is always good to normalise your data after figuring out the mean and standard
# deviations of its channels.
x_train = datasets.MNIST(
    root="datasets/",  # saves dataset in sub-folder
    train=True,  # get training dataset from MNIST
    transform=transforms.ToTensor(),  # transforms dataset to tensor
    download=True,  # downloads if dataset isn't available locally
)
# Set batches after shuffling; this is actually an epoch of training on its own
# Shuffling is usually good: for instance, with the MNIST dataset you don't want the model to process a batch only of
# ones or twos, but rather a mix of them. However, in the case of time series data, where the order is important,
# shuffling can be bad.
train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)

x_test = datasets.MNIST(
    root="datasets/",
    train=False,  # get testing dataset from MNIST
    transform=transforms.ToTensor(),
    download=True,
)
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=True)

# initialise nn
model = NN(input_size, n_classes).to(
    device
)  # always set to device to take advantage of cuda!

# loss and optimiser
# one mustn't use softmax in their networks when employing CrossEntropyLoss() in calculating loss, as the function
# already performs a softmax itself. Doubling the softmax could lead to the "vanishing gradients" problem.
calculate_loss = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# train nn

for epoch in range(n_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        # printing data.shape would give [64, 1, 28, 28] => [batch_size, channels, image height, image width]
        # channels are the sources of information; MNIST is black-and-white, so it has just 1 channel
        # we'd like to get the data in the form [64, 784], so we unroll the last three indices in the vector
        data = data.reshape(
            data.shape[0],  # preserves first dimension
            -1,  # flattens remaining dimensions
        )

        # forward
        scores = model(data)  # gives shape [64, 10], same as targets
        loss = calculate_loss(scores, targets)

        # backward
        # Backpropagation is the stage where the model gets a suggestion of what to do next time at that layer. By
        # default, gradients are accumulated; we don't want that here, we want a clean slate each time, or else each
        # decision would be influenced by the past and the model would slow down in attaining information from new
        # batches, severely affecting accuracy.
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        optimiser.zero_grad()
        loss.backward()

        # optimisation
        # The optimisation step makes a decision based on suggestions from the backpropagation stage. It updates
        # weights depending on the gradients passed by loss.backward().
        optimiser.step()  # updates weights depending on gradients from loss.backward()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data:")
    else:
        print("Checking accuracy on testing data.")
    n_correct = 0
    n_samples = 0
    # move from training mode to evaluation mode
    # when evaluating you want certain elements such as batchnorms or dropouts to be turned off, which are used to
    # prevent overfitting and are not valuable in an accuracy check here.
    # see: https://stackoverflow.com/questions/44223585/why-disable-dropout-during-validation-and-testing
    # https://stackoverflow.com/questions/45497342/batch-normalization-during-testing
    # Using PyTorch Lightning makes this step redundant.
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

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
