import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class BengaliDatasetTrain(Dataset):
    def __init__(self,
                 folds,
                 aug=None,
                 preload=False,
                 RGB=True):
        """ Initialize the BengaliDatasetTrain dataset

        Args:
            - folds: number of folds to use
            - aug: albumentations
            - preload: if preload the dataset into memory
            - RGB: convert image to RGB
        """
        self.folds = folds
        self.images = None
        self.aug = aug
        self.RGB = RGB

        df = pd.read_csv('data/train_folds.csv')
        df = df[df.kfold.isin(folds)].reset_index(drop=True)

        df = df[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'kfold']].reset_index(drop=True)

        self.image_id = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values

        # preload dataset into memory
        if preload:
            self._preload()

    def _preload(self):
        """
        Preload images to memory
        """
        self.images = []

        print('Preloading data...')
        for image_fn in tqdm(self.image_id, total=len(self.image_id)):
            # load images
            image = Image.open(f'data/train_images/{image_fn}.png')
            self.images.append(image.copy())
            # avoid too many opened files bug
            image.close()

    # probably the most important to customize.
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
        else:
            # If on-demand data loading
            image = Image.open(f'data/train_images/{self.image_id[index]}.png')

        if self.RGB:
            image = image.convert('RGB')

        image = np.array(image)

        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.aug is not None:
            image = self.aug(image=image)['image']

        image = image.astype(np.float32)

        if self.RGB:
            image = image.transpose((2, 0, 1))
        else:
            image = image[np.newaxis, :]

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'grapheme_root': torch.tensor(self.grapheme_root[index], dtype=torch.long),
            'vowel_diacritic': torch.tensor(self.vowel_diacritic[index], dtype=torch.long),
            'consonant_diacritic': torch.tensor(self.consonant_diacritic[index], dtype=torch.long)
        }

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.image_id)
