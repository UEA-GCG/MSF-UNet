import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

def get_patch(img, lr, gt, scale, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))
    lr_tx = tx // scale
    lr_ty = ty // scale
    lr_tp = tp // scale

    return img[ty:ty + tp, tx:tx + tp], lr[lr_ty:lr_ty + lr_tp, lr_tx:lr_tx + lr_tp], gt[ty:ty + tp, tx:tx + tp]

class RGBDD_Dataset(Dataset):
    """RGB-D-D Dataset."""

    def __init__(self, root_dir, scale=4, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            downsample (str): kernel type of downsample, real mean use real LR and HR data
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        types = ['models', 'plants', 'portraits']

        self.transform = transform
        self.scale = scale

        self.GTs = []
        self.RGBs = []

        list_dir = os.listdir(root_dir)
        for name in list_dir:
            if name.find('RGB') > -1:
                self.RGBs.append('%s/%s' % (root_dir, name))
            elif name.find('HR_gt') > -1:
                self.GTs.append('%s/%s' % (root_dir, name))
        self.RGBs.sort()
        self.GTs.sort()

    def __len__(self):
        return len(self.GTs)

    def __getitem__(self, idx):

        image = Image.open(self.RGBs[idx]).convert("RGB")
        image = np.array(image).astype(np.float32)
        gt = Image.open(self.GTs[idx])
        w, h = gt.size
        s = self.scale
        lr = np.array(gt.resize((w//s,h//s),Image.BICUBIC)).astype(np.float32)
        gt = np.array(gt).astype(np.float32)

        # normalization
        maxx = np.max(lr)
        minn = np.min(lr)
        lr=(lr-minn)/(maxx-minn)

        image_max = np.max(image)
        image_min = np.min(image)
        image = (image-image_min)/(image_max-image_min)


        if self.transform:
            image = self.transform(image).float()
            gt = self.transform(np.expand_dims(gt,2)).float()
            lr = self.transform(np.expand_dims(lr,2)).float()

        sample = {'guidance': image, 'lr': lr, 'gt': gt, 'max': maxx, 'min': minn}

        return sample
















# from torchvision import transforms
# import numpy as np
# import os
# import random
#
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
#
#
# def modcrop(image, modulo):
#     h, w = image.shape[0], image.shape[1]
#     h = h - h % modulo
#     w = w - w % modulo
#
#     return image[:h, :w]
#
#
# class RGBDD_Dataset(Dataset):
#     """RGB-D-D Dataset."""
#
#     def __init__(self, root_dir, scale=4, transform=None):
#         """
#         Args:
#             root_dir (string): Directory with all the images.
#             scale (float): dataset scale
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#
#         self.transform = transform
#         self.scale = scale
#
#         self.GTs = []
#         self.RGBs = []
#
#         list_dir = os.listdir(root_dir)
#         for name in list_dir:
#             if name.find('RGB') > -1:
#                 self.RGBs.append('%s/%s' % (root_dir, name))
#             elif name.find('HR_gt') > -1:
#                 self.GTs.append('%s/%s' % (root_dir, name))
#         self.RGBs.sort()
#         self.GTs.sort()
#
#     def __len__(self):
#         return len(self.GTs)
#
#     def __getitem__(self, idx):
#         image = np.array(Image.open(self.RGBs[idx]))
#         gt = np.array(Image.open(self.GTs[idx]))
#         assert gt.shape[0] == image.shape[0] and gt.shape[1] == image.shape[1]
#         s = self.scale
#         image = modcrop(image, s)
#         gt = modcrop(gt, s)
#
#         h, w = gt.shape[0], gt.shape[1]
#         s = self.scale
#
#         lr = np.array(Image.fromarray(gt).resize((w // s, h // s), Image.BICUBIC)).astype(np.float32)
#         gt = gt / 255.0
#         image = image / 255.0
#         lr = lr / 255.0
#
#         if self.transform:
#             image = self.transform(image).float()
#             gt = self.transform(np.expand_dims(gt, 2))
#             lr = self.transform(np.expand_dims(lr, 2)).float()
#
#         # sample = {'guidance': image, 'lr': lr, 'gt': gt, 'max':maxx, 'min': minn}
#         sample = {'guidance': image, 'lr': lr, 'gt': gt}
#         return sample