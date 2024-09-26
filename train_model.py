import os
import subprocess


def create_dataset(path_home, path_images, path_dataset):
    """
    Function to create a dataset for StyleGAN2-ADA using the dataset_tool.py script.

    Parameters:
    - path_home: Path to the home directory where the stylegan2-ada-pytorch repo is located
    - path_images: Path to the directory containing the images
    - path_dataset: Path to the directory where the dataset should be saved

    Returns:
    - None
    """
    # Build the command for creating the dataset
    cmd = f"python {path_home}/stylegan2-ada-pytorch/dataset_tool.py --source {path_images} --dest {path_dataset}"
    
    # Print the command (for debugging)
    print(f"Creating dataset with command: {cmd}")
    
    try:
        # Run the command using subprocess
        subprocess.run(cmd, shell=True, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error while running the command: {e}")



def run_stylegan_training(path_home, path_exp, path_dataset, snap=10):
    """
    Function to run the StyleGAN2-ADA training with the specified parameters.

    Parameters:
    - path_dataset: Path to the dataset directory
    - path_exp: Path to the experiments directory (output)
    - path_home: Path to the home directory where the stylegan2-ada-pytorch repo is located
    - snap: Snapshot interval (default is 10)

    Returns:
    - None
    """
    # Build the command for running StyleGAN training
    cmd = f"python {path_home}/stylegan2-ada-pytorch/train.py "\
          f"--snap {snap} --cond=1 --outdir {path_exp} --data {path_dataset}"
    
    # Print the command (for debugging)
    print(f"Running command: {cmd}")
    
    # try:
    #     # Run the command using subprocess
    #     subprocess.run(cmd, shell=True, check=True)
        
    # except subprocess.CalledProcessError as e:
    #     print(f"Error while running the command: {e}")
        

# Example usage:
# Assuming you have set up the required paths:
# path_dataset = "/home/user/dataset"
# path_exp = "/home/user/experiments"
# path_home = "/home/user/stylegan2-ada-pytorch"

# run_stylegan_training(path_dataset, path_exp, path_home, snap=10)




def resume_stylegan_training(path_dataset, path_exp, path_home, network_snapshot, experiment_folder, snap=10):
    """
    Function to resume StyleGAN2-ADA training from a snapshot (.pkl) file.

    Parameters:
    - path_dataset: Path to the dataset directory
    - path_exp: Path to the experiments directory (output)
    - path_home: Path to the home directory where the stylegan2-ada-pytorch repo is located
    - network_snapshot: The .pkl file (network snapshot) to resume training from
    - experiment_folder: Folder containing the previous experiment
    - snap: Snapshot interval (default is 10)

    Returns:
    - None
    """
    
    # Build the full path to the .pkl file
    resume_path = os.path.join(path_exp, experiment_folder, network_snapshot)
    
    # Build the command to resume training from the .pkl file
    cmd = f"python {path_home}/stylegan2-ada-pytorch/train.py "\
          f"--snap {snap} --resume {resume_path} --outdir {path_exp} --data {path_dataset}"
    
    # Print the command (for debugging)
    print(f"Running command: {cmd}")
    
    # try:
    #     # Run the command using subprocess
    #     subprocess.run(cmd, shell=True, check=True)
        
    # except subprocess.CalledProcessError as e:
    #     print(f"Error while running the command: {e}")

# Example usage:
# Assuming you have set up the required paths and values:
# path_dataset = "/path/to/dataset"
# path_exp = "/path/to/experiments"
# path_home = "/path/to/stylegan2-ada-pytorch"
# network_snapshot = "network-snapshot-000100.pkl"
# experiment_folder = "00008-circuit-auto1-resumecustom"

# resume_stylegan_training(path_dataset, path_exp, path_home, network_snapshot, experiment_folder, snap=10)
