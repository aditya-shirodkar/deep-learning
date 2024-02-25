# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
# Data augmentation generates more data for training purposes. This can be done by using transforms.
# Working with the kaggle cats and dogs dataset: https://www.kaggle.com/c/dogs-vs-cats/data
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from customDatasetLoader import ImageClassifierDatasetLoader


# All operations are done in sequence. We will use this sequence to inflate our dataset.
# Always consider the kind of data you are using. For instance, if using pictures of digits like in the MNIST dataset,
# vertical or horizontal flips may not make sense. Here, with cats and dogs, horizontal flips are fine but vertical
# ones still dodgy.
my_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),  # the transforms we need for augmentation apply on PIL images
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),  # randomly crops to these dimensions
        transforms.ColorJitter(brightness=0.5),  # brightness changes to the image
        transforms.RandomHorizontalFlip(p=0.5),  # p -> probability of the flip
        transforms.RandomVerticalFlip(p=0.025),  # unlikely to see these
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # here we need to first find the mean and standard deviations across all training examples to normalise them
        # it will subtract the mean from the channel value for all examples and then divide by the standard deviation
        # in this case it does nothing!
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]
)
dataset = ImageClassifierDatasetLoader(
    csv_file="datasets/kaggle_cats_dogs/train.csv",
    root_dir="datasets/kaggle_cats_dogs/train",
    transform=my_transforms,
)

# We're saving the files here so you can inspect the augmentations. THIS WILL GENERATE A LOT OF IMAGES.
# However, you don't actually need to save any of the transformed images; you could just pass them instead to the data
# loader. Furthermore, you'll have to remake a CSV file accounting for the newly generated images as done before.
path_to_save = "datasets/kaggle_cats_dogs/train_augmented/"
cat_num = 0
dog_num = 0
for i in range(10):  # 10 new images will be created per sample
    for img, label in dataset:
        if label == 0:  # if it's a cat
            save_image(img, path_to_save + "cat." + str(cat_num) + ".jpeg")
            cat_num += 1
        else:
            save_image(img, path_to_save + "dog." + str(dog_num) + ".jpeg")
            dog_num += 1
