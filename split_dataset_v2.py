import json
import pandas as pd
from collections import Counter
import random
from tqdm import tqdm

def split_train_test(json_file, train_ratio, seed=None):
    """
    Split the data from a JSON file into train and test sets based on the train_ratio.
    Returns train_df and test_df DataFrames.
    """
    # Load data from JSON file
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

    # Create DataFrame
    df = pd.DataFrame(labels, columns=['file_path', 'label'])

    if seed:
        random.seed(seed)

    class_counts = Counter(df['label'])
    num_classes = len(class_counts)
    class_subset_sizes = {class_id: int(subset_size / num_classes) for class_id in class_counts}

    subsets = []
    for class_id, count in tqdm(class_counts.items(), desc='Subsetting classes', unit='class'):
        class_df = df[df['label'] == class_id]
        class_subset = class_df.sample(n=class_subset_sizes[class_id], replace=False, random_state=seed)
        subsets.append(class_subset)

    subset_df = pd.concat(subsets, ignore_index=True)
    return subset_df

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




