import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import argparse
import subprocess


parser = argparse.ArgumentParser(description='Partitions the data downloaded from https://www.kaggle.com/c/diabetic-retinopathy-detection/data for the experiments')
parser.add_argument('csv_table', type=Path, help='CSV file containing image names and associated labels')
parser.add_argument('images_folder', type=Path, help='Path to the folder containing all original images')
parser.add_argument('transformed_images_folder', type=Path, help='Path to the folder to save the transformed 128x128 images to')
parser.add_argument('partitions_folder', type=Path, help='Path to the folder where the partitions will be saved using symbolic links')
args = parser.parse_args()

n_partitions = 30
test_size = 0.2

df = pd.read_csv(args.csv_table)
n_classes = len(df['level'].unique())

args.transformed_images_folder.mkdir(exist_ok=True)
print('Resizing images...')
subprocess.run(['magick', 'mogrify', '-resize', 'x128', '-path', str(args.transformed_images_folder), str(args.images_folder / '*.jpeg')])
print('Cropping images...')
subprocess.run(['magick', 'mogrify', '-crop', '1:1', '-gravity', 'Center', str(args.transformed_images_folder / '*.jpeg')])

args.partitions_folder.mkdir(exist_ok=True)
for partition in range(n_partitions):
    for c in range(n_classes):
        (args.partitions_folder / f'{partition}/train/{c}').mkdir(parents=True, exist_ok=True)
        (args.partitions_folder / f'{partition}/test/{c}').mkdir(parents=True, exist_ok=True)

sss = StratifiedShuffleSplit(n_partitions, test_size=test_size, random_state=0)
for partition, (train_index, test_index) in tqdm(enumerate(sss.split(df, df['level']))):
    train_df: pd.DataFrame = df.iloc[train_index]  # type:ignore
    test_df: pd.DataFrame = df.iloc[test_index]  # type:ignore
    for name, partition_df in zip(('train', 'test'), (train_df, test_df)):
        for row in tqdm(partition_df.itertuples(), total=len(partition_df), leave=False, desc=name):
            image_path = args.transformed_images_folder / f'{row.image}.jpeg'
            assert image_path.is_file()
            new_path = args.partitions_folder / f'{partition}/{name}/{row.level}/{image_path.name}'
            if not new_path.exists():
                print(f'{new_path} -> {image_path}')
                new_path.symlink_to(image_path.resolve())
