import glob

import pandas as pd
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    files = glob.glob('data/train_image_data_*.parquet')

    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop('image_id', axis=1)
        images = df.values.reshape(-1, 137, 236)

        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            img = Image.fromarray(images[j])
            img.save(f'data/train_images/{img_id}.png')
