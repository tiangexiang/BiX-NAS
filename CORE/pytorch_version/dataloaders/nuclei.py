from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from PIL import Image
import numpy as np
import torch
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt

class Nuclei(Dataset):
    # load all data into cpu, consistent to keras version.
    def __init__(self, path, data_name, batchsize=1, steps=None, shuffle=False, transforms=None):
        self.x, self.y = self.load_data(path, data_name)
        self.transforms = transforms
        self.steps = steps
        # self.CLASS_WEIGHTS = torch.tensor([1., 1.]).cuda() # use cuda
        self.CLASS_WEIGHTS = torch.tensor([1., 1.]) # use cpu
        # if steps is not None:
        #     self.idx_mapping = np.random.randint(0, self.x.shape[0], steps*batchsize)
        #     self.steps = self.steps * batchsize

    def __len__(self):
        # return self.steps if self.steps is not None else self.x.shape[0]
        return self.x.shape[0]
    
    def __getitem__(self, index):
        # if self.steps is not None:
        #     index = self.idx_mapping[index]

        if self.transforms is not None:
            x, y = self.transforms(images=self.x[None, index], segmentation_maps=self.y[None, index])
        else:
            x, y = self.x[None, index], self.y[None, index]

        x, y = x.astype('float32')[0]/255., y.astype('float32')[0]/255.
        x, y = ToTensor()(x), ToTensor()(y)
        ret = {}
        ret['x'] = x.float()
        ret['y'] = y.float()
        return ret

    def load_data(self, path, data_name):
        # imgaug requires 0-255 unit8 images
        print('loading '+data_name+' data...')
        if data_name == 'monuseg':
            folders = os.listdir(path)
            imgs_list = []
            masks_list = []
            for folder in folders:
                items = os.listdir(os.path.join(path,'x'))
                for item in items:
                    imgs_list.append(np.array(Image.open(os.path.join(path,'x',item)).convert('RGB')))
                    masks_list.append(np.array(Image.open(os.path.join(path,'y',item)).convert('L')))
        elif data_name == 'tnbc':
            folders = os.listdir(path)

            mask_config = []
            image_config = []
            for folder in folders:
                a,b = list(folder.split('_'))
                
                items = os.listdir(os.path.join(path,folder))
                for item in items:
                    entry = []
                    fnum, idx = list(item.split('_'))
                    fnum = int(fnum)
                    idx = int(idx[:-4])
                    entry.append(fnum)
                    entry.append(idx)
                    entry.append(os.path.join(path,folder,item))
                    if a[:2] == 'GT':
                        mask_config.append(entry)
                    else:
                        image_config.append(entry)

            image_config = sorted(image_config, key = lambda x : (x[0],x[1]))
            mask_config = sorted(mask_config, key = lambda x : (x[0],x[1]))

            imgs_list = []
            masks_list = []
            for i in range(len(image_config)):
                imgs_list.append(np.array(Image.open(image_config[i][-1]).convert('RGB')))
                masks_list.append(np.array(Image.open(mask_config[i][-1])))
        else:
            print('dataset name cannot be recognized! Program terminating...')
            exit()

        imgs_np = np.asarray(imgs_list)
        masks_np = np.asarray(masks_list)

        x = np.asarray(imgs_np, dtype=np.uint8)
        y = np.asarray(masks_np, dtype=np.uint8)
        
        if len(y.shape) == 3:
            y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
            
        print("Successfully loaded data from "+path)
        print("data shape:", x.shape,y.shape)
        
        return x, y