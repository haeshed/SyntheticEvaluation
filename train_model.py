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
    
    print(f"Creating dataset with command: {cmd}")
    
    try:
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
    
    print(f"Running command: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error while running the command: {e}")
        

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
    
    print(f"Running command: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error while running the command: {e}")



def generate_stylegan_images(path_home, path_model, path_out, seeds_num):
    """
    Function to generate images using the StyleGAN2-ADA model for the CIFAR-10 dataset.

    Parameters:
    - path_home: Path to the home directory where the stylegan2-ada-pytorch repo is located
    - path_out: Path to the output directory
    - seeds_num: Range of seeds to use for image generation (e.g., "0-35")
    - classes: Classes to generate images for (e.g., "0-9" for all classes)

    Returns:
    - None
    """
    

    for class_num in range(10):
        class_num = int(class_num)
        class_out_dir = os.path.join(path_out, f"class_{class_num}")
        os.makedirs(class_out_dir, exist_ok=True)

        # Build the command for generating images with StyleGAN2-ADA
        cmd = f"python {path_home}/stylegan2-ada-pytorch/generate.py " \
              f"--outdir={class_out_dir} --seeds={seeds_num} --class={class_num} " \
              f"--network={path_model}"

        print(f"Running command: {cmd}")

        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Images for class {class_num} generated successfully in {class_out_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error while generating images for class {class_num}: {e}")



