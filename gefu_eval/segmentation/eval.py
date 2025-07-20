"""
Author: Rundong Luo, rundongluo2002@gmail.com
"""

import os
from utils.helpers import get_test_trans, get_test_trans_ms
from utils.routines import eval_evaluate
from datasets.cityscapes import CityscapesExt
from datasets.nighttime import NighttimeDataset
from models.refinenet import RefineNet

import torch
import torch.nn as nn

def main(args):
    # Define data transformation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    target_size = (360,640)
    test_trans = get_test_trans_ms(mean, std, target_size)

    # Load dataset
    testset_nd = NighttimeDataset("data/bdd100k-night", transforms=test_trans)

    dataloaders = {}
    dataloaders['test_nd'] = torch.utils.data.DataLoader(testset_nd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    num_classes = len(CityscapesExt.validClasses)

    criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
    model = RefineNet(num_classes, pretrained=False)

    # Push model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    # Load weights from checkpoint
    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    print('--- eval - Nighttime ---')
    test_acc_nd, test_loss_nd, miou_nd, confmat_nd, iousum_nd = eval_evaluate(dataloaders['test_nd'],
        model, criterion, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std, save_root=args.save_path, save=args.save, ms=True)
    print(miou_nd)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation evaluation')
    parser.add_argument('--weight', type=str, default='checkpoints/best_weights.pth.tar',
                        help='load weight file')
    parser.add_argument('--model', type=str, default='refinenet')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 3)')
    parser.add_argument('--workers', type=int, default=8, metavar='W',
                        help='number of data workers (default: 4)')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--save_path', default='./',type=str)
    parser.add_argument('--save', action='store_true', help='save visual results')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    main(args)
