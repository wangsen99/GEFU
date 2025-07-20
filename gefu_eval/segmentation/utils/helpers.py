import torch
import torchvision.transforms.functional as TF

import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageFilter
from time import localtime, strftime
from copy import deepcopy

################################
### Data helpers
################################

def get_test_trans_ms(mean, std, target_size):
    # Return test transform function
    def test_trans_ms(image, mask=None):
        # Basic image pre-processing
        image = TF.resize(image, target_size, interpolation=Image.LANCZOS) # Resize, 1 for LANCZOS, 2 for BILINEAR
        w, h = image.size
        # From PIL to Tensor

        multiscale = [0.5, 0.75, 1]
        # multiscale = [1]
        img_resized_list = []
        for scale in multiscale:
            target_h = int(h * scale)
            target_w = int(w * scale)
            img_resized = image.resize((target_w, target_h), Image.LANCZOS)
            img_resized = TF.to_tensor(img_resized)
            img_resized_list.append(img_resized)

        image = TF.to_tensor(image)
        
        if mean and std:
            image = TF.normalize(image, mean, std) # Normalize
        
        if mask:
            mask = TF.resize(mask, target_size, interpolation=Image.NEAREST) # 0 for Image.NEAREST
            mask = np.array(mask, np.uint8) # PIL Image to numpy array
            mask = torch.from_numpy(mask) # Numpy array to tensor
            return image, mask, img_resized_list
        else:
            return image

    return test_trans_ms

def get_test_trans(mean, std, target_size):
    # Return test transform function
    def test_trans(image, mask=None):
        # Basic image pre-processing
        image = TF.resize(image, target_size, interpolation=Image.LANCZOS) # Resize, 1 for LANCZOS, 2 for BILINEAR

        # From PIL to Tensor
        image = TF.to_tensor(image)
        
        if mean and std:
            image = TF.normalize(image, mean, std) # Normalize
        
        if mask:
            mask = TF.resize(mask, target_size, interpolation=Image.NEAREST) # 0 for Image.NEAREST
            mask = np.array(mask, np.uint8) # PIL Image to numpy array
            mask = torch.from_numpy(mask) # Numpy array to tensor
            return image, mask
        else:
            return image

    return test_trans

def visim(img, mean=None, std=None):
    img = img.cpu()
    # Convert image data to visual representation
    if std:
        img *= torch.tensor(std)[:,None,None]
    if mean:
        img += torch.tensor(mean)[:,None,None]
    npimg = (img.numpy()*255).astype('uint8')
    if len(npimg.shape) == 3 and npimg.shape[0] == 3:
        npimg = np.transpose(npimg, (1, 2, 0))
    else:
        npimg = npimg[0,:,:]
    return npimg
    
def vislbl(label, mask_colors):
    label = label.cpu()
    # Convert label data to visual representation
    label = np.array(label.numpy())
    if label.shape[-1] == 1:
        label = label[:,:,0]
    
    # Convert train_ids to colors
    label = mask_colors[label]
    return label

################################
### Generate run directories
################################
def gen_train_dirs(name):
    """
    Generate directory structure for storing files produced during training run.
    """
    # date_time = strftime("%Y%m%d-%H%M%S", localtime())
    # Define paths
    run_dir = os.path.join('runs',name)
    logs_dir = os.path.join(run_dir,'logs')
    settings_path = os.path.join(run_dir,'settings.txt')
    weight_dir = os.path.join(run_dir,'weights')
    img_dir = os.path.join(run_dir,'images')
    # Create dirs
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        os.makedirs(weight_dir)
        os.makedirs(img_dir)
        os.makedirs(logs_dir)
    # Set pwd to run dir
    os.chdir(run_dir)

def gen_test_dirs(name):
    """
    Generate directory structure for storing files produced during training run.
    """
    # date_time = strftime("%Y%m%d-%H%M%S", localtime())
    # Define paths
    run_dir = os.path.join('runs-loss',name)
    logs_dir = os.path.join(run_dir,'logs')
    settings_path = os.path.join(run_dir,'settings.txt')
    weight_dir = os.path.join(run_dir,'weights')
    img_dir = os.path.join(run_dir,'images')
    # Create dirs
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        os.makedirs(weight_dir)
        os.makedirs(img_dir)
        os.makedirs(logs_dir)
    # Set pwd to run dir
    os.chdir(run_dir)

def gen_eval_dirs():
    """
    Generate directory structure for storing files produced during evaluation run.
    """
    date_time = strftime("%Y%m%d-%H%M%S", localtime())
    # Generate run dir
    run_dir = os.path.join('evals',date_time)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir

################################
### Evaluation
################################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          figsize=(12,12)):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.seterr(divide='ignore', invalid='ignore')
    
    if not title:
        title = 'Confusion matrix'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.set_size_inches(figsize)
    return fig

################################
### Progress meters
################################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def cutout_multi(img1, img2, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img1 = np.array(img1)
        img2 = np.array(img2)
        mask = np.array(mask)

        img_h, img_w, img_c = img1.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            if erase_w < img_w and erase_h < img_h:
                x = np.random.randint(0, img_w - erase_w)
                y = np.random.randint(0, img_h - erase_h)
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img1[y:y + erase_h, x:x + erase_w, :] = value
        img2[y:y + erase_h, x:x + erase_w, :] = value
        mask[y:y + erase_h, x:x + erase_w] = 0

        img1 = Image.fromarray(img1.astype(np.uint8))
        img2 = Image.fromarray(img2.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img1, img2, mask