import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import argparse
import subprocess


parser = argparse.ArgumentParser(description='Partitions the Adience dataset downloaded from https://talhassner.github.io/home/projects/Adience/Adience-data.html'
                                             'for the experiments. Discards incorrectly labelled data.')
parser.add_argument('folds_folder', type=Path, help='Folder containing the original fold data (fold_0_data.txt - fold_4_data.txt)')
parser.add_argument('images_folder', type=Path, help='Path to the folder containing the decompressed aligned.tar.gz file')
parser.add_argument('transformed_images_folder', type=Path, help='Path to the folder to save the transformed 256x256 images to')
parser.add_argument('partitions_folder', type=Path, help='Path to the folder where the partitions will be saved using symbolic links')
args = parser.parse_args()

n_partitions = 30
test_size = 0.2

ranges = [
    (0, 2),
    (4, 6),
    (8, 13),
    (15, 20),
    (25, 32),
    (38, 43),
    (48, 53),
    (60, 100),
]

folds = [pd.read_csv(args.folds_folder / f'fold_{f}_data.txt', sep='\t') for f in range(5)]

def assign_range(age: str):
    age = eval(age)
    
    if age is None:
        return None
    
    if age in ranges:
        return ranges.index(age)
    
    if isinstance(age, tuple):
        age_minimum, age_maximum = age
        for i, (range_minimum, range_maximum) in enumerate(ranges):
            if (age_minimum >= range_minimum) and (age_maximum <= range_maximum):
                return i
        return None
    
    if isinstance(age, int):
        for i, (range_minimum, range_maximum) in enumerate(ranges):
            if (age >= range_minimum) and (age <= range_maximum):
                return i
        return None
    
    return None

def image_path_from_row(row):
    return f'{row["user_id"]}/landmark_aligned_face.{row["face_id"]}.{row["original_image"]}'

fold_dfs = list()
for f, fold in enumerate(folds):
    fold = fold.assign(age=fold['age'].map(assign_range))
    notna = fold['age'].notna()
    n_discarded = (~notna).sum()
    print(f'Fold {f}: discarding {n_discarded} entries ({(n_discarded / len(fold)) * 100:.1f}%)')
    fold = fold.loc[notna]
    fold = fold.assign(age=fold['age'].astype(int))
    
    fold_dfs.append(pd.DataFrame(dict(path=fold.apply(image_path_from_row, axis='columns'), age=fold['age'])))
df: pd.DataFrame = pd.concat(fold_dfs, ignore_index=True)  # type: ignore

args.transformed_images_folder.mkdir(exist_ok=True)
print('Resizing images...')
for row in tqdm(df.itertuples(), total=len(df)):
    dst_image = args.transformed_images_folder / row.path
    if dst_image.is_file():
        continue
    src_image = args.images_folder / row.path
    dst_image.parent.mkdir(exist_ok=True, parents=True)
    subprocess.run(['convert', '-resize', 'x128', str(src_image), str(dst_image)])

args.partitions_folder.mkdir(exist_ok=True)
for partition in range(n_partitions):
    for c in range(len(ranges)):
        (args.partitions_folder / f'{partition}/train/{c}').mkdir(parents=True, exist_ok=True)
        (args.partitions_folder / f'{partition}/test/{c}').mkdir(parents=True, exist_ok=True)
        
sss = StratifiedShuffleSplit(n_partitions, test_size=test_size, random_state=0)
for partition, (train_index, test_index) in tqdm(enumerate(sss.split(df, df['age']))):
    train_df: pd.DataFrame = df.iloc[train_index]  # type: ignore
    test_df: pd.DataFrame = df.iloc[test_index]  # type: ignore
    for name, partition_df in zip(('train', 'test'), (train_df, test_df)):
        for row in tqdm(partition_df.itertuples(), total=len(partition_df), leave=False, desc=name):
            image_path = args.transformed_images_folder / row.path
            assert image_path.is_file()
            new_path = args.partitions_folder / f'{partition}/{name}/{row.age}/{image_path.name}'
            if not new_path.exists():
                new_path.symlink_to(image_path.resolve())
