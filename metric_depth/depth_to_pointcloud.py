"""
Born out of Depth Anything V1 Issue 36
Make sure you have the necessary libraries installed.
Code by @1ssb

This script processes a set of images to generate depth maps and corresponding point clouds.
The resulting point clouds are saved in the specified output directory.

Usage:
    python script.py --encoder vitl --load-from path_to_model --max-depth 20 --img-path path_to_images --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4

Arguments:
    --encoder: Model encoder to use. Choices are ['vits', 'vitb', 'vitl', 'vitg'].
    --load-from: Path to the pre-trained model weights.
    --max-depth: Maximum depth value for the depth map.
    --img-path: Path to the input image or directory containing images.
    --outdir: Directory to save the output point clouds.
    --focal-length-x: Focal length along the x-axis.
    --focal-length-y: Focal length along the y-axis.
"""

import argparse
import cv2
import numpy as np
import os
from PIL import Image
import torch
import json
import re
import csv

from depth_anything_v2.dpt import DepthAnythingV2

def crop_rgb_image(image, target_size=(518, 518)):
    """
    Crops an RGB image to a specified square size by adding black bars
    and then performing a center crop.

    Args:
        image (numpy.ndarray): A NumPy array representing the RGB image to crop.
            The expected shape is (height, width, 3), where the last dimension
            represents the RGB channels.
        target_size (tuple, optional): The desired square size of the cropped
            image (height, width). Defaults to (518, 518).  Both values must be the same.

    Returns:
        numpy.ndarray: The cropped RGB image with the specified square size.
    """
    height, width, channels = image.shape

    if height > width:
        raise ValueError("Height cannot be greater than width.  Padding logic assumes width is the larger dimension.")
    if channels != 3:
        raise ValueError(f"Expected an RGB image with 3 channels, but got {channels}")
    if target_size[0] != target_size[1]:
        raise ValueError(f"Target size must be square (height == width), but got {target_size}")
    target_height = target_size[0]
    target_width  = target_size[1]


    padding_needed = target_width - height
    padding_top = padding_needed // 2
    padding_bottom = padding_needed - padding_top

    # Create black bars (arrays of zeros with the same width and 3 channels)
    black_bar_top = np.zeros((padding_top, width, 3), dtype=image.dtype)
    black_bar_bottom = np.zeros((padding_bottom, width, 3), dtype=image.dtype)

    # Add the black bars to the original image array
    padded_image_array = np.concatenate((black_bar_top, image, black_bar_bottom), axis=0)

    padded_height, padded_width, _ = padded_image_array.shape

    # Calculate the cropping coordinates to get a square center crop
    crop_size = target_size[0]
    start_row = (padded_height - crop_size) // 2
    end_row = start_row + crop_size
    start_col = (padded_width - crop_size) // 2
    end_col = start_col + crop_size

    # Crop the image
    cropped_image_array = padded_image_array[start_row:end_row, start_col:end_col, :]
    return cropped_image_array



def de_crop_rgb_image(cropped_image, original_height, original_width):
    """
    Reverses the crop_rgb_image function. It takes a cropped RGB image
    and returns an image with the original dimensions by padding
    it with black bars.

    Args:
        cropped_image (numpy.ndarray): The cropped RGB image (assumed to be square).
            Expected shape is (crop_size, crop_size, 3).
        original_height (int): The original height of the image *before* cropping.
        original_width (int): The original width of the image *before* cropping.

    Returns:
        numpy.ndarray: The "de-cropped" RGB image with the original dimensions.
                         The image will have black bars at the top and bottom.
    """
    crop_size, _, channels = cropped_image.shape
    if crop_size != 518:
        raise ValueError(f"Expected cropped image size to be 518x518, but got {crop_size}x{crop_size}")

    if channels != 3:
        raise ValueError(f"Expected cropped image to have 3 channels, but got {channels}")
    # Calculate padding needed to get back to the padded size *before* cropping.
    padded_height_before_crop = crop_size + ((original_width - crop_size) // 2) * 2

    # Create an array of zeros with the padded height and width
    padded_image = np.zeros((padded_height_before_crop, original_width, 3), dtype=cropped_image.dtype)

    # Calculate where to place the cropped image in the larger image
    start_row = (padded_height_before_crop - crop_size) // 2
    end_row = start_row + crop_size
    start_col = (original_width - crop_size) // 2
    end_col = start_col + crop_size

    # Place the cropped image in the center of the larger image
    padded_image[start_row:end_row, start_col:end_col, :] = cropped_image

    # Calculate padding to remove
    padding_needed = padded_image.shape[0] - original_height
    padding_top = padding_needed // 2
    padding_bottom = padding_needed - padding_top
    # Remove the top and bottom padding
    original_image = padded_image[padding_top:padded_image.shape[0]-padding_bottom, :, :]

    return original_image

def cropImage(image):
    
    height, width= image.shape
    target_height = width
    padding_needed = target_height - height
    padding_top = padding_needed // 2
    padding_bottom = padding_needed - padding_top

    black_bar_top = np.zeros((padding_top, width), dtype=image.dtype)
    black_bar_bottom = np.zeros((padding_bottom, width), dtype=image.dtype)

    padded_image_array = np.concatenate((black_bar_top, image, black_bar_bottom), axis=0)
    padded_height, padded_width = padded_image_array.shape
    # Calculate the cropping coordinates to get a 518x518 center crop
    crop_size = 518
    start_row = (padded_height - crop_size) // 2
    end_row = start_row + crop_size
    start_col = (padded_width - crop_size) // 2
    end_col = start_col + crop_size

    # Crop the image
    cropped_image_array = padded_image_array[start_row:end_row, start_col:end_col]
    return cropped_image_array

def deCropImage(cropped_image, original_height, original_width):
    """
    Reverses the cropImage function.  It takes a cropped image (assumed to be
    518x518) and returns an image with the original dimensions by padding
    it with black bars.

    Args:
        cropped_image (numpy.ndarray): The cropped image (assumed to be 518x518).
        original_height (int): The original height of the image *before* cropping.
        original_width (int):         # Read the image using OpenCV
The original width of the image *before* cropping.

    Returns:
        numpy.ndarray: The "de-cropped" image with the original dimensions.
                         The image will have black bars at the top and bottom.
    """
    if cropped_image.shape != (518, 518):
        raise ValueError(f"Expected cropped image shape (518, 518), but got {cropped_image.shape}")

    # Calculate padding needed to get back to the padded size *before* cropping.
    padded_height_before_crop = 518 + ((original_width - 518) // 2) * 2

    # Create an array of zeros with the padded height and width
    padded_image = np.zeros((padded_height_before_crop, original_width), dtype=cropped_image.dtype)

    # Calculate where to place the cropped image in the larger image
    start_row = (padded_height_before_crop - 518) // 2
    end_row = start_row + 518
    start_col = (original_width - 518) // 2 # same as in cropImage
    end_col = start_col + 518

    # Place the cropped image in the center of the larger image
    padded_image[start_row:end_row, start_col:end_col] = cropped_image

    # Calculate padding to remove
    padding_needed = padded_image.shape[0] - original_height
    padding_top = padding_needed // 2
    padding_bottom = padding_needed - padding_top
    # Remove the top and bottom padding
    original_image = padded_image[padding_top:padded_image.shape[0]-padding_bottom, :]

    return original_image

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate point clouds from images.')
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to use.')
    parser.add_argument('--load-from', default='', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--max-depth', default=20, type=float,
                        help='Maximum depth value for the depth map.')
    parser.add_argument('--img-path', type=str, required=True,
                        help='Path to the input image or directory containing images.')
    parser.add_argument('--outdir', type=str, default='./vis_pointcloud',
                        help='Directory to save the output point clouds.')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')

    args = parser.parse_args()

    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    #depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    model_state_dict = torch.load(args.load_from, map_location='cpu')['model']
    model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    depth_anything.load_state_dict(model_state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get the list of image files to process
    #if os.path.isfile(args.img_path):
    #    if args.img_path.endswith('txt'):
    #        with open(args.img_path, 'r') as f:
    #            filenames = f.read().splitlines()
    #    else:
    #        filenames = [args.img_path]
    #else:
    #    filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    # Create the output directory if it doesn't exist
    #os.makedirs(args.outdir, exist_ok=True)

    with open(args.img_path, 'r') as f:
        filelist = f.read().splitlines()
        #header=filelist
        dataset_path=filelist[0]
        sonar_configuration = json.load(open(dataset_path+'sonar-configuration.json'))
        sonar_model=sonar_configuration["P900"]
        filelist.pop(0)

    # Process each image file
    for k, filename in enumerate(filelist):
        print(f'Processing {k+1}/{len(filelist)}: {filename}')
        integers = [int(s) for s in re.findall(r'\d+', filename)]
        
        MISSION=integers[0]
        AUV=integers[2]
        INDEX=integers[3]
        """ 
        # Load the image
        image = Image.open(filename).convert('L')
        image=np.array(image)
        image=cropImage(image)

        image=image.reshape(1,518,518,1)
        #_, height = image.size
        """

        # Load the image
        #image_pil = Image.open(filename).convert('RGB')
        #image_np = np.array(image_pil,dtype=np.uint8)
        #original_height, original_width,_ = image_np.shape

        # Crop the grayscale image
        #cropped_image_np = crop_rgb_image(image_np)
        #resized_image_np = np.resize(image_np, (518, 518)) # Simple resize for now
        #image=cropped_image_np.reshape(3,518,518).astype(np.uint8)

        raw_image = cv2.imread(filename)


        pred = depth_anything.infer_image(raw_image, 518)
        #print("#################\n",pred.min(),"#################\n")
        # De-crop the depth prediction
        #pred = de_crop_rgb_image(pred, original_height, original_width)

        with open(f"{dataset_path}mission{MISSION}.csv", newline='') as f:
            reader = csv.reader(f)
            mission_metadata = list(reader)
            mission_metadata.pop(0)

        mission=mission_metadata[AUV]

        #pred = depth_anything.infer_image(image, 518)
        #pred=deCropImage(pred,original_height,sonar_model["RangeBins"],original_width)
        auv_metadata=json.load(open(f"/home/guilherme/Documents/SEE-Dataset/Sonar-Dataset-mission-{MISSION}-P900/auv-{AUV}/Meta-data/{INDEX}.json"))
        #pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255.0
        #pred = pred.astype(np.uint8)
        
        #radius, theta = pred.shape
        cv2.imshow(pred)
        print(pred)
        """
        try:
            with open(args.outdir, 'a') as outfile:
                for t in range(theta):
                    for r in range(radius):
                        if pred[r][t]>0:
                            
                            rad = (r*sonar_model["RangeMax"])/sonar_model["RangeBins"] + sonar_model["RangeMin"]
                            phi_= (pred[r][t]-(sonar_model["Elevation"]/2))
                            theta_= ((t*(sonar_model["Azimuth"])/theta)-sonar_model["Azimuth"]/2) +auv_metadata["yaw"]
                            
                            
                            x=(rad*np.cos(np.deg2rad(theta_))*np.cos(np.deg2rad(phi_)))+(auv_metadata["x"]-float(mission[2]))
                            y=(rad*np.sin(np.deg2rad(theta_))*np.cos(np.deg2rad(phi_)))+(auv_metadata["y"]+float(mission[3]))
                            z=(rad*np.sin(np.deg2rad(phi_))) + auv_metadata["z"]
                            outfile.write(f"{x} {y} {z}\n")
        except:
            pass
        """

if __name__ == '__main__':
    main()
