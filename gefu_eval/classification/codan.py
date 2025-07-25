from PIL import Image
import os
import os.path
import numpy as np
import tarfile
import torchvision.transforms as transforms

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import extract_archive


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class CODaN_withPath(VisionDataset):
    """`CODaN <https://github.com/Attila94/CODaN>`_ Dataset.

    Args:
        data (string, optional): Location of the downloaded .tar.bz2 files.
        split (string, optional): Define which dataset split to use. Must be one of
            'train', 'val', 'test_day', 'test_night'.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """


    def __init__(self, root='./', split='train', transform=None, target_transform=None):

        super(CODaN_withPath, self).__init__(root, transform, target_transform)

        cls_list = ['Bicycle', 'Car', 'Motorbike', 'Bus', 'Boat', 'Cat', 'Dog', 'Bottle', 'Cup', 'Chair']
        # split_list = ['train','val','test_day','test_night','trainval','train_night','train_day_night','test_night_full','test_night_half','train_night2day_synthesized_224']
        # assert split in split_list, 'Invalid split.'

        self.split = split  # dataset split
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        self.paths = []

        # Unpack archives
        if not os.path.isdir(os.path.join(root,'data',split)):
            # Join .tar.bz2 parts files for training split
            if split == 'train' and not os.path.exists(os.path.join(root,'data','codan_train.tar.bz2')):
                with open(os.path.join(root,'data','codan_train.tar.bz2'), 'wb') as f_out:
                    for i in range(3):
                        fpath = os.path.join(root,'data','codan_train.tar.bz2.part{}'.format(i))
                        with open(fpath, 'rb') as f_in:
                            f_out.write(f_in.read())
                        os.remove(fpath)
            # Unpack tar
            tarpath = os.path.join(root,'data/codan_'+split+'.tar.bz2')
            with tarfile.open(tarpath) as tar:
                print('Unpacking {} split.'.format(split))
                tar.extractall(path=os.path.join(root,'data'))
        else:
            print('Loading CODaN {} split...'.format(split))

        # loop through split directory, load all images in memory using PIL
        for i, c in enumerate(cls_list):
            im_dir = os.path.join(root,'data',split,c)
            ims = os.listdir(im_dir)
            ims = [im for im in ims if is_image_file(im)] # remove any system files
            
            for im in ims:
                path = os.path.join(im_dir,im)
                img = Image.open(path)
                self.paths.append(path)
                self.data.append(img.copy())
                img.close()
                self.targets.append(i)
        print('Dataset {} split loaded.'.format(split), 'Number of images:', len(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, path = self.data[index], self.targets[index], self.paths[index]
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path


    def __len__(self):
        return len(self.data)

class CODaN(VisionDataset):
    """`CODaN <https://github.com/Attila94/CODaN>`_ Dataset.

    Args:
        data (string, optional): Location of the downloaded .tar.bz2 files.
        split (string, optional): Define which dataset split to use. Must be one of
            'train', 'val', 'test_day', 'test_night'.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """


    def __init__(self, root='./', split='train', transform=None, target_transform=None):

        super(CODaN, self).__init__(root, transform, target_transform)

        cls_list = ['Bicycle', 'Car', 'Motorbike', 'Bus', 'Boat', 'Cat', 'Dog', 'Bottle', 'Cup', 'Chair']
        self.split = split  # dataset split
        self.data = []
        self.targets = []
        self.paths = []
        self.transform = transform
        self.target_transform = target_transform
        for i, c in enumerate(cls_list):
            im_dir = os.path.join(root,'data',split,c)
            ims = os.listdir(im_dir)
            ims = [im for im in ims if is_image_file(im)] # remove any system files
            
            for im in ims:
                img = Image.open(os.path.join(im_dir,im))
                self.data.append(img.copy())
                img.close()
                self.paths.append(os.path.join(im_dir,im))
                self.targets.append(i)
        print('Dataset {} split loaded.'.format(split), 'Number of images:', len(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)
