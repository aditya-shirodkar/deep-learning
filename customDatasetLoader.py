# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
import os
import pandas as pd
from PIL import Image  # used for image -> image operations
from skimage import io  # used to read images in numerical form for ML stuff
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence  # to pad batches
from torch.utils.data import Dataset


# expecting csv file with image filename in the first column and integer classification label in the next
class ImageClassifierDatasetLoader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    # required for the torch.utils.data.DataLoader() function
    # failing to do so will result in a NotImplementedError
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0])
        img = io.imread(img_path)
        y_label = torch.tensor(int(self.df.iloc[index, 1]))

        if self.transform:
            img = self.transform(img)

        return img, y_label


# load the tokeniser vocabulary used by spacy; check out https://spacy.io/models
# python -m spacy download en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        #  a frequency threshold is the minimum repetitions of a word/ngram for our model to capture it as significant
        # tokens: padding; start-of-sentence; end-of-sentence; unknown
        self.index_to_str = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.str_to_index = {v: k for k, v in self.index_to_str.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.index_to_str)

    # basic tokeniser which separates by spaces
    @staticmethod  # to avoid setting a "self"
    def tokeniser(text):
        return [t.text.lower() for t in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        words_index = 4  # first four indices set above

        for sentence in sentence_list:
            for word in self.tokeniser(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.str_to_index[word] = words_index
                    self.index_to_str[words_index] = word
                    words_index += 1

    def numericalise(self, text):
        tokenised_text = self.tokeniser(text)

        # return a list of only those characters which are already in self.str_to_index, having passed the frequency
        # threshold test above
        return [
            self.str_to_index[t]
            if t in self.str_to_index
            else self.str_to_index["<UNK>"]
            for t in tokenised_text
        ]


# requires a folder of images (to root_dir) and a csv with two columns titled "image" and "caption"
# "image" column would contain image filenames, and "caption" brief descriptions of these images
class ImageToTextClassifierDatasetLoader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, freq_threshold=5):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # initialise vocabulary and build vocabulary for current dataset
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numerical_caption = [self.vocab.str_to_index["<SOS>"]]
        numerical_caption += self.vocab.numericalise(caption)
        numerical_caption.append(self.vocab.str_to_index["<EOS>"])

        return img, torch.tensor(numerical_caption)


class Collater:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        # unsqueezing to add an extra dimension to store the batch
        # works as: https://stackoverflow.com/a/65831759
        # concatenating all images in a batch into the newly created 0th dimension
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)  # assumes images are of the same size
        targets = [item[1] for item in batch]
        # targets (captions) must be of the same size, hence:
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_index)

        return imgs, targets
