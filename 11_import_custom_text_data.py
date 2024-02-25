# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
# To load text, using something like PyTorch Text is often adequate. However, for some cases like image captioning
# building one's own custom loader is best.
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from customDatasetLoader import ImageToTextClassifierDatasetLoader, Collater


# goals:
# convert text to numerical values:
# 1) vocabulary to map each word/ngram to an index
# 2) set up a PyTorch dataset to load the data -> does the numeric conversion
# 3) set up padding for every batch -> so all examples are of the same sequence length -> set up data loader
def get_loader(
    csv_file,
    root_dir,
    transform,
    batch_size=32,
    n_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = ImageToTextClassifierDatasetLoader(
        csv_file, root_dir, transform=transform
    )
    pad_index = dataset.vocab.str_to_index["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collater(pad_index=pad_index),
    )

    return loader


def main():
    # resizing images to be of the same size and then converting them to tensors
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    dataLoader = get_loader(
        csv_file="datasets/kaggle_flickr_8k/captions.txt",
        root_dir="datasets/kaggle_flickr_8k/Images",
        transform=transform,
    )

    for idx, (imgs, captions) in enumerate(dataLoader):
        print(imgs.shape)
        print(captions.shape)


if __name__ == "__main__":
    main()
