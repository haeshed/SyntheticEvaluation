import os
import subprocess

def generate_synthetic_images(outdir, seeds, class_range, network_checkpoint, generate_script_path):
    """
    Function to generate synthetic images using StyleGAN2-ADA from a given model checkpoint.

    Parameters:
    - outdir: Directory where the output images will be saved.
    - seeds: Range of seeds for generating images (e.g., "0-35").
    - class_range: Range of class labels for conditional generation (e.g., "0-3" for classes 0, 1, 2, 3).
    - network_checkpoint: Path to the network checkpoint .pkl file (can be a local path or a URL).
    - generate_script_path: Path to the `generate.py` script in the stylegan2-ada-pytorch repo.

    Returns:
    - None
    """
    
    # Determine the number of images to generate
    seed_start, seed_end = map(int, seeds.split('-'))
    total_images = seed_end - seed_start + 1
    
    # Determine class range
    class_start, class_end = map(int, class_range.split('-'))
    classes = list(range(class_start, class_end + 1))
    num_classes = len(classes)

    # Calculate the number of images to generate per class
    images_per_class = total_images // num_classes
    
    # Generate images for each class
    for cls in classes:
        cls_seeds = f"{seed_start}-{seed_start + images_per_class - 1}"  # Adjust seeds for the current class
        cmd = f"python {generate_script_path} --outdir={outdir} --seeds={cls_seeds} --class={cls} --network={network_checkpoint}"
        
        # Print the command for debugging
        print(f"Running command for class {cls}: {cmd}")
        
        try:
            # Run the command using subprocess
            subprocess.run(cmd, shell=True, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Error while running the command for class {cls}: {e}")

# Example usage:
# outdir = "out"
# seeds = "0-35"
# class_range = "0-3"  # Generate images for classes 0, 1, 2, 3
# network_checkpoint = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl"
# generate_script_path = "/path/to/stylegan2-ada-pytorch/generate.py"

# generate_synthetic_images(outdir, seeds, class_range, network_checkpoint, generate_script_path)
