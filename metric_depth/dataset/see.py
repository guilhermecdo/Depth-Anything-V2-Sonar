import numpy as np
import json
import math
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

class SEE(Dataset):
    def __init__(self,filelist_path,mode:str,sonar_model:str="P900"):
        self.batch=1
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        self.dataset_path=self.filelist[0]
        sonar_configuration = json.load(open( self.dataset_path+'sonar-configuration.json'))
        self.sonar_model=sonar_configuration[sonar_model]
        self.filelist.pop(0)

    def cropImage(self,image):
        height, width = image.shape
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
    
    def grayscale_to_rgb(self,grayscale_image):

        if len(grayscale_image.shape) != 2:
            raise ValueError("Input image must be a 2D array (grayscale)")

        height, width = grayscale_image.shape
        rgb_image = np.zeros((height, width, 3), dtype=grayscale_image.dtype)

        # Replicate the grayscale values across the three RGB channels
        rgb_image[:, :, 0] = grayscale_image  # Red channel
        rgb_image[:, :, 1] = grayscale_image  # Green channel
        rgb_image[:, :, 2] = grayscale_image  # Blue channel

        return rgb_image

    def __getitem__(self, item):
        img_path=self.filelist[item].split(' ')[1]
        pl_path=self.filelist[item].split(' ')[0]
        image=np.load(img_path)
        image=self.cropImage(image)
        image=image.reshape(518,518)
        image=self.grayscale_to_rgb(image)
        image=image.reshape(3,518,518)
        
        gt_mask=np.load(pl_path)
        theta, phi = gt_mask.shape
        gt=np.zeros(shape=(self.sonar_model["RangeBins"],theta),dtype=np.uint8)
        
        #for b in range(self.batch):
        for t in range(theta):
            for p in range(phi):
                r_index = int(math.floor(((gt_mask[t][p]-(self.sonar_model["RangeMin"]))*self.sonar_model["RangeBins"])/self.sonar_model["RangeMax"]))
                gt[r_index][t]=p
        gt=self.cropImage(gt)
        #gt=gt.reshape(1,518,518)
        sample={'image':torch.from_numpy(image),'depth':torch.from_numpy(gt)}
        sample['valid_mask'] = sample['depth']
        
        return sample

    def __len__(self):
        return len(self.filelist)