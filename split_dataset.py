# train/test split

import json
import pandas as pd
from collections import Counter
import random
from tqdm import tqdm

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['labels']

def create_dataframe(data):
    df = pd.DataFrame(data, columns=['file_path', 'label'])
    return df

def split_data(df, train_ratio, seed=None):
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

def save_data(data_df, output_file):
    data = {
        "labels": data_df.values.tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
input_file = raw_data + "/dataset.json"
# output_file = raw_data + "/dataset_subset.json"
output_dir = raw_data

# input_file = 'path/to/input.json'
# output_dir = 'path/to/output_dir'
train_ratio = 0.8  # 80% for train, 20% for test
seed_value = 42

print(f"Loading data from {input_file}")
data = load_data(input_file)
df = create_dataframe(data)

print(f"Splitting data into train and test sets with train ratio {train_ratio} and seed {seed_value}")
train_df, test_df = split_data(df, train_ratio, seed=seed_value)

train_output_file = f"{output_dir}/train_data.json"
test_output_file = f"{output_dir}/test_data.json"

print(f"Saving train data to {train_output_file}")
save_data(train_df, train_output_file)

print(f"Saving test data to {test_output_file}")
save_data(test_df, test_output_file)

print("Train data statistics:")
print(train_df['label'].value_counts().sort_index())

print("Test data statistics:")
print(test_df['label'].value_counts().sort_index())



# Split into specific subset size

import json
import pandas as pd
from collections import Counter
import random
from tqdm import tqdm

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['labels']

def create_dataframe(data):
    df = pd.DataFrame(data, columns=['file_path', 'label'])
    return df

def subset_data(df, subset_size, seed=None):
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

def save_data(subset_df, output_file):
    data = {
        "labels": subset_df.values.tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
subset_size = 4000
seed_value = 42
input_file = raw_data + "/dataset.json"
# output_file = raw_data + "/dataset_subset.json"
output_file = raw_data + f"/dataset_subset_size{subset_size}_seed{seed_value}.json"

print(f"Loading data from {input_file}")
data = load_data(input_file)
df = create_dataframe(data)

print(f"Subsetting data with size {subset_size} and seed {seed_value}")
subset_df = subset_data(df, subset_size, seed=seed_value)

print(f"Saving data to {output_file}")
save_data(subset_df, output_file)

print("Subset statistics:")
print(subset_df['label'].value_counts().sort_index())