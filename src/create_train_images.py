import argparse
import glob
import os

import pandas as pd
from PIL import Image
import joblib
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create train images')
parser.add_argument('--format', '-f', default='png', choices=['png', 'pkl'],
                    help='format of files: (default: png)')


def encode_png(img_array, img_id):
    img = Image.fromarray(img_array)
    img.save(f'data/train_images_png/{img_id}.png')


def encode_pkl(img_array, img_id):
    joblib.dump(img_array, f'data/train_images_pkl/{img_id}.pkl')


encoders = {
    'png': encode_png,
    'pkl': encode_pkl
}


def main():
    args = parser.parse_args()

    if not os.path.isdir(f'data/train_images_{args.format}'):
        os.mkdir(f'data/train_images_{args.format}')

    encoder = encoders[args.format]
    files = glob.glob('data/train_image_data_*.parquet')

    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop('image_id', axis=1)
        images = df.values.reshape(-1, 137, 236)

        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            encoder(images[j], img_id)


if __name__ == '__main__':
    main()
