import argparse
import os
import json

def parse_args():
    desc = "Tool to create multiclass json labels file for stylegan2-ada-pytorch"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--verbose', action='store_true',
                        help='Print progress to console.')

    parser.add_argument('--input_folder', type=str,
                        default='./input/',
                        help='Directory path to the inputs folder. (default: %(default)s)')

    parser.add_argument('--output_folder', type=str,
                        default='./output/',
                        help='Directory path to the outputs folder. (default: %(default)s)')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Create output directory if it does not exist
    remakePath = args.output_folder
    if not os.path.exists(remakePath):
        os.makedirs(remakePath)

    data_dict = {'labels': []}
    image_counter = 0
    label_dict = {}

    # Open the JSON file for writing
    with open(os.path.join(remakePath, 'dataset.json'), 'w') as outfile:
        for root, subdirs, files in os.walk(args.input_folder):
            if len(subdirs) > 0:
                # Process subdirectories and create label dictionary
                for subdir in subdirs:
                    label_name = os.path.basename(subdir)
                    label_dict[subdir] = label_name
                    if args.verbose:
                        print(f'Found class directory: {label_name}')

            if len(files) > 0:
                current_subdir = os.path.basename(root)
                if current_subdir in label_dict:
                    label_name = label_dict[current_subdir]
                    for filename in files:
                        file_path = os.path.join(current_subdir, filename)
                        data_dict['labels'].append([file_path, label_name])
                        image_counter += 1

                        if args.verbose:
                            if image_counter % 1000 == 0:
                                print(f'Processed {image_counter} images.')

        # Final output statistics
        if args.verbose:
            print(f'Processing complete. Total images processed: {image_counter}')
            print(f'Total classes (directories): {len(label_dict)}')

        # Write the collected data to the JSON file
        json.dump(data_dict, outfile, indent=4)
        print(f'JSON file saved to {os.path.join(remakePath, "dataset.json")}')

if __name__ == "__main__":
    main()
