import json
import pandas as pd
from collections import Counter
import random
from tqdm import tqdm

def load_json_data(json_file):
    """
    Load data from a JSON file with the structure {'labels': [...]}
    Returns a list of [file_path, label] pairs.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['labels']

def create_dataframe(data):
    """
    Convert the data list to a pandas DataFrame with columns 'file_path' and 'label'.
    """
    df = pd.DataFrame(data, columns=['file_path', 'label'])
    return df

def split_data(df, train_ratio, seed=None):
    """
    Split the data into train and test sets based on the train_ratio.
    Returns train_df and test_df DataFrames.
    """
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

def subset_data(df, subset_size, seed=None):
    """
    Create a subset of the data with the specified size, ensuring an even distribution of classes.
    Returns a subset DataFrame.
    """
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
    print(data_df['label'].value_counts().sort_index())

def main(input_file, output_dir, train_ratio=0.8, subset_size=None, seed=42):
    """
    Main function to handle data splitting and subsetting.
    """
    print(f"Loading data from {input_file}")
    data = load_json_data(input_file)
    df = create_dataframe(data)

    if subset_size is not None:
        print(f"Subsetting data with size {subset_size} and seed {seed}")
        subset_df = subset_data(df, subset_size, seed=seed)
        output_file = f"{output_dir}/dataset_subset_size{subset_size}_seed{seed}.json"
        print(f"Saving subset data to {output_file}")
        save_data(subset_df, output_file)
        print_class_distribution(subset_df, "Subset")
    else:
        print(f"Splitting data into train and test sets with train ratio {train_ratio} and seed {seed}")
        train_df, test_df = split_data(df, train_ratio, seed=seed)

        train_output_file = f"{output_dir}/train_data.json"
        test_output_file = f"{output_dir}/test_data.json"

        print(f"Saving train data to {train_output_file}")
        save_data(train_df, train_output_file)

        print(f"Saving test data to {test_output_file}")
        save_data(test_df, test_output_file)

        print_class_distribution(train_df, "Train")
        print_class_distribution(test_df, "Test")

# Example usage
input_file = "/path/to/input.json"
output_dir = "/path/to/output_dir"

# For train/test split
# main(input_file, output_dir, train_ratio=0.8, seed=42)

# For subsetting
main(input_file, output_dir, subset_size=4000, seed=42)