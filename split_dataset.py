import json
import pandas as pd
from collections import Counter
import random
from tqdm import tqdm
import shutil
import os
import numpy as np
import glob


def split_train_test(json_file, train_ratio, seed=None):
    """
    Split the data from a JSON file into train and test sets based on the train_ratio.
    Returns train_df and test_df DataFrames.
    """
    # Load data from JSON fileכה
    with open(json_file, 'r') as f:
        data = json.load(f)
    labels = data['labels']

    # Create DataFrame
    df = pd.DataFrame(labels, columns=['file_path', 'label'])

    if seed:
        random.seed(seed)

    # Split the data into train and test sets
    grouped = df.groupby('label')
    train_data = []
    test_data = []

    for _, group in tqdm(grouped, desc='Splitting data', unit='class'):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the group
        train_size = int(len(group) * train_ratio)
        train_data.extend(group.iloc[:train_size].to_dict('records'))
        test_data.extend(group.iloc[train_size:].to_dict('records'))

    train_df = pd.DataFrame(train_data, columns=['file_path', 'label'])
    test_df = pd.DataFrame(test_data, columns=['file_path', 'label'])

    return train_df, test_df

def subset_data(json_file, subset_size, seed=None):
    """
    Create a subset of the data from a JSON file with the specified size, ensuring an even distribution of classes.
    Returns a subset DataFrame.
    """
    # Load data from JSON file
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
        # class_subset = class_df.sample(n=subset_size, replace=False, random_state=seed)
        # print('len(class_df), subset_size, len(class_counts): ', len(class_df), subset_size, len(class_counts))
        class_subset = class_df.sample(n=min(count, subset_size), replace=False, random_state=seed)
        subsets.append(class_subset)

    subset_df = pd.concat(subsets, ignore_index=True)
    return subset_df


def copy_images_to_model_and_dataset(input_df ,input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print(input_df)
    for index, row in input_df.iterrows():
        # Construct the source and destination paths
        src_path = os.path.join(input_dir, row['file_path'])
        dest_path = os.path.join(output_dir, os.path.basename(row['file_path']))
        
        # Copy the image to the destination folder
        shutil.copy(src_path, dest_path)
        
    # save_data(input_df, output_dir + "/dataset.json")



def save_data(data_df, output_file):
    """
    Save the data DataFrame to a JSON file with the structure {'labels': [...]}
    """
    data = {
        "labels": data_df.values.tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def print_class_distribution(data_df, name):
    """
    Print the class distribution in the provided DataFrame.
    """
    print(f"{name} data statistics:")
    print(f"Total number of samples: {len(data_df)}")
    print(data_df['label'].value_counts().sort_index())


def open_folders(model_name, base_dir):
    experiments_dir = os.path.join(base_dir, 'experiments')
    dataset_dir = os.path.join(base_dir, 'dataset')
    images_dir = os.path.join(base_dir, 'images')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    return dataset_dir, experiments_dir, images_dir



def generate_labels_json(path_all_images, path_output, output_file_name):
    print(f"Starting to generate labels JSON file...")
    print(f"Base directory: {path_all_images}")

    labels_data = {"labels": []}
    file_count = 0

    # Walk through the base directory and its subdirectories
    for root, dirs, files in os.walk(path_all_images):
        for file in files:
            # Only include image files (you can adjust this based on file extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                # Extract the class label from the first digit of the file name
                try:
                    label = int(file[0])  # First digit of the file name
                except ValueError:
                    # Skip files that do not start with a digit
                    continue

                # Create relative file path from the base directory
                file_path = os.path.relpath(os.path.join(root, file), path_all_images)

                # Add the file path and corresponding label to the list
                labels_data["labels"].append([file_path, label])
                file_count += 1

            # Print progress after every 1000 files
                if file_count % 10000 == 0:
                    print(f"Processed {file_count} files...")

    os.makedirs(path_output, exist_ok=True)
    output_file_path = os.path.join(path_output, output_file_name)

    # Write the labels to the output JSON file
    with open(output_file_path, 'w') as f:
        json.dump(labels_data, f, indent=4)

    print(f"Generated labels JSON file with {len(labels_data['labels'])} entries.")



def distribute_files_to_label_dirs(src_dir):
    # Create class directories if they don't exist
    for class_num in range(10):
        class_dir = os.path.join(src_dir, f"{class_num}")
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    # Distribute files into their respective class directories
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        if os.path.isfile(src_path):
            try:
                file_class = filename.split("_")[0][0]
                if file_class.isdigit():
                    dst_path = os.path.join(src_dir, f"{file_class}", filename)
                    shutil.move(src_path, dst_path)
            except (IndexError, ValueError):
                # Handle files that don't follow the expected naming convention
                print(f"Skipping file: {filename} (doesn't follow the expected naming convention)")



def delete_images_and_dataset_dirs(parent_dir):
    # Define the subdirectories and the file to move
    subdirs_to_delete = ['images', 'dataset']
    file_to_move = 'dataset.json'
    
    # Move the dataset.json file if it exists
    images_dir = os.path.join(parent_dir, 'images')
    file_path = os.path.join(images_dir, file_to_move)
    
    if os.path.exists(file_path):
        # Move the file to the parent directory
        shutil.move(file_path, os.path.join(parent_dir, file_to_move))
        print(f"Moved: {file_path} to {parent_dir}")
    else:
        print(f"File does not exist: {file_path}")
    
    # Delete the specified subdirectories
    for subdir in subdirs_to_delete:
        dir_path = os.path.join(parent_dir, subdir)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)  # Recursively delete the directory and its contents
            print(f"Deleted: {dir_path}")
        else:
            print(f"Directory does not exist: {dir_path}")




def get_latest_pkl_file(dir_path):
    # Use glob to find all .pkl files in the directory and its subdirectories
    pkl_files = glob.glob(os.path.join(dir_path, '**', '*.pkl'), recursive=True)
    
    if not pkl_files:
        print("No .pkl files found.")
        return None

    # Get the most recent file based on the modification time
    latest_file = max(pkl_files, key=os.path.getmtime)
    
    return latest_file
