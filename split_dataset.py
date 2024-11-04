import json
import pandas as pd
from collections import Counter
import random
from tqdm import tqdm
import shutil
import os
import numpy as np
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_train_test(json_file, train_ratio, seed=None):
    logger.info(f"Splitting data from {json_file} with train ratio: {train_ratio}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    labels = data['labels']

    df = pd.DataFrame(labels, columns=['file_path', 'label'])

    if seed:
        random.seed(seed)

    grouped = df.groupby('label')
    train_data = []
    test_data = []

    for _, group in tqdm(grouped, desc='Splitting data', unit='class'):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_size = int(len(group) * train_ratio)
        train_data.extend(group.iloc[:train_size].to_dict('records'))
        test_data.extend(group.iloc[train_size:].to_dict('records'))

    train_df = pd.DataFrame(train_data, columns=['file_path', 'label'])
    test_df = pd.DataFrame(test_data, columns=['file_path', 'label'])

    logger.info(f"Split completed: {len(train_df)} training samples and {len(test_df)} testing samples.")
    return train_df, test_df

def subset_data(json_file, subset_size, seed=None):
    logger.info(f"Creating subset from {json_file} with size: {subset_size}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    labels = data['labels']

    df = pd.DataFrame(labels, columns=['file_path', 'label'])

    if seed:
        random.seed(seed)

    class_counts = Counter(df['label'])
    subset_size = int(subset_size / len(class_counts))
    subsets = []
    
    for class_id, count in tqdm(class_counts.items(), desc='Subsetting classes', unit='class'):
        class_df = df[df['label'] == class_id]
        class_subset = class_df.sample(n=min(count, subset_size), replace=False, random_state=seed)
        subsets.append(class_subset)

    subset_df = pd.concat(subsets, ignore_index=True)
    logger.info(f"Subset creation completed with {len(subset_df)} total samples.")
    return subset_df

def copy_images_to_model_and_dataset(input_df, input_dir, output_dir):
    logger.info(f"Copying images from {input_dir} to {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for index, row in input_df.iterrows():
        src_path = os.path.join(input_dir, row['file_path'])
        dest_path = os.path.join(output_dir, os.path.basename(row['file_path']))
        
        try:
            shutil.copy(src_path, dest_path)
            logger.debug(f"Copied {src_path} to {dest_path}")
        except Exception as e:
            logger.error(f"Error copying {src_path} to {dest_path}: {e}")

def save_data(data_df, output_file):
    logger.info(f"Saving data to {output_file}")
    
    data = {
        "labels": data_df.values.tolist()
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Data saved successfully: {output_file}")

def print_class_distribution(data_df, name):
    logger.info(f"{name} data statistics:")
    logger.info(f"Total number of samples: {len(data_df)}")
    logger.info(f"Class distribution:\n{data_df['label'].value_counts().sort_index()}")

def open_folders(model_name, base_dir):
    logger.info(f"Creating directories for model: {model_name} in {base_dir}")
    
    experiments_dir = os.path.join(base_dir, 'experiments')
    dataset_dir = os.path.join(base_dir, 'dataset')
    images_dir = os.path.join(base_dir, 'images')
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    logger.info(f"Directories created: {dataset_dir}, {experiments_dir}, {images_dir}")
    return dataset_dir, experiments_dir, images_dir

def generate_labels_json(path_all_images, path_output, output_file_name):
    logger.info(f"Starting to generate labels JSON file...")
    logger.info(f"Base directory: {path_all_images}")

    labels_data = {"labels": []}
    file_count = 0

    for root, dirs, files in os.walk(path_all_images):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                try:
                    label = int(file[0])
                except ValueError:
                    logger.warning(f"Skipping file: {file} (does not start with a digit)")
                    continue

                file_path = os.path.relpath(os.path.join(root, file), path_all_images)
                labels_data["labels"].append([file_path, label])
                file_count += 1

                if file_count % 10000 == 0:
                    logger.info(f"Processed {file_count} files...")

    os.makedirs(path_output, exist_ok=True)
    output_file_path = os.path.join(path_output, output_file_name)

    with open(output_file_path, 'w') as f:
        json.dump(labels_data, f, indent=4)

    logger.info(f"Generated labels JSON file with {len(labels_data['labels'])} entries.")

def distribute_files_to_label_dirs(src_dir):
    logger.info(f"Distributing files in {src_dir} to label directories.")
    
    for class_num in range(10):
        class_dir = os.path.join(src_dir, f"{class_num}")
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        if os.path.isfile(src_path):
            try:
                file_class = filename.split("_")[0][0]
                if file_class.isdigit():
                    dst_path = os.path.join(src_dir, f"{file_class}", filename)
                    shutil.move(src_path, dst_path)
                    logger.debug(f"Moved {src_path} to {dst_path}")
            except (IndexError, ValueError):
                logger.warning(f"Skipping file: {filename} (doesn't follow the expected naming convention)")

def delete_images_and_dataset_dirs(parent_dir):
    logger.info(f"Deleting images and dataset directories in {parent_dir}")
    
    subdirs_to_delete = ['images', 'dataset']
    file_to_move = 'dataset.json'
    
    images_dir = os.path.join(parent_dir, 'images')
    file_path = os.path.join(images_dir, file_to_move)
    
    if os.path.exists(file_path):
        shutil.move(file_path, os.path.join(parent_dir, file_to_move))
        logger.info(f"Moved: {file_path} to {parent_dir}")
    else:
        logger.warning(f"File does not exist: {file_path}")
    
    for subdir in subdirs_to_delete:
        dir_path = os.path.join(parent_dir, subdir)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"Deleted: {dir_path}")
        else:
            logger.warning(f"Directory does not exist: {dir_path}")

def get_latest_pkl_file(dir_path):
    logger.info(f"Searching for the latest .pkl file in {dir_path}")
    
    pkl_files = glob.glob(os.path.join(dir_path, '**', '*.pkl'), recursive=True)
    
    if not pkl_files:
        logger.warning("No .pkl files found.")
        return None

    latest_file = max(pkl_files, key=os.path.getmtime)
    logger.info(f"Latest .pkl file found: {latest_file}")
    
    return latest_file
