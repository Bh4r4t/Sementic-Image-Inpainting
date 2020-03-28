import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os

class PatchDataset(Dataset):
    """dataset for inpainting"""
    def __init__(self, root, img_size, transform, mask_type, args):
        """
        Args:
            root (string): path to the image folder.
            img_size (number): size of the images expected.
            tranform : transform to be applied on images. 
            mask_type (optional) : center or random.
        """
        self.root = root
        self.img_size = img_size
        self.transform = transform
        self.imgs = os.listdir(root)
        self.mask_type = mask_type
        self.args = args
        
    def __len__(self):
        return len(self.imgs)
    

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.imgs[idx]))
        image = self.transform(image)
        
        #mask
        mask = torch.ones(image.size(), dtype = torch.float64)
        
        if self.mask_type == 'center':
            h = image.size(1)//4
            w = image.size(2)//4
            mask[:, h: image.size(1)-h, w:image.size(2)-h] = 0.0
            
        elif self.mask_type == 'random':
            mid_x = np.random.randint(image.size(1)//4, image.size(1)//2)
            mid_y = np.random.randint(image.size(2)//4, image.size(2)//2)
            h = np.random.randint(image.size(1)//5, 8*image.size(1)//10)
            w = np.random.randint(image.size(2)//5, 8*image.size(2)//10)
            
            h_begin, h_end = max(0, h-mid_x//2), min(image.size(1), h+mid_x//2)
            w_begin, w_end = max(0, w-mid_y//2), min(image.size(2), h+mid_y//2)
            
            mask[:, h_begin:h_end, w_begin:w_end] = 0.0
        
        weighted_mask = weightedMask(mask, self.args)
        mask = mask.unsqueeze(0)
        corrupted_img = image.clone()
        corrupted_img = corrupted_img*mask
        
        return self.imgs[idx], corrupted_img, image, mask, weighted_mask
    
    
    
def weightedMask(mask, args):
    """
    Args:
        mask: patch mask of an image
        ndim: size of the kernal for weighted term of a pixel
    """
    kernel = torch.ones((3, 3, args.kernel_size, args.kernel_size), dtype=torch.float64)/(args.kernel_size**2)
    inv_mask = (1 - mask).unsqueeze(0)
    res = F.conv2d(inv_mask, kernel, padding=3)
    res = (mask*res).squeeze(0)
    return res
