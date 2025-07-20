import torch
import time
import os
import cv2
from utils.helpers import AverageMeter, ProgressMeter, visim, vislbl
from utils.get_iou import iouCalc
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import torch.distributed as dist
import torchvision.transforms.functional as TF
import torchvision
import copy
import random
import torch.nn.functional as F

def reduce_mean(tensor, nprocs):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= nprocs
	return rt

def eval_evaluate(dataloader, model, criterion, classLabels, validClasses, void=-1, maskColors=None, mean=None, std=None, save_root='',save_suffix='', save=False, ms=False):
	save_path = os.path.join('results', save_root)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	paths = [os.path.join(save_path,'visuals')]

	for p in paths:
		if not os.path.exists(p):
			os.makedirs(p)

	iou = iouCalc(classLabels, validClasses, voidClass = void)
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	loss_running = AverageMeter('Loss', ':.4e')
	acc_running = AverageMeter('Acc', ':6.2f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, loss_running, acc_running],
		prefix='Test: ')
	
	# set model in training mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for epoch_step, (inputs, labels, filepath, ms_img) in enumerate(dataloader):
			data_time.update(time.time()-end)

			# input resolution
			h,w = labels.shape[-2:]
			res = h*w
			size = labels.size()[-2:]
			
			inputs = inputs.float().cuda()
			labels = labels.long().cuda()
			scores = torch.zeros(1, 20, labels.shape[1], labels.shape[2]).cuda().detach()
	
			# forward
			flip = False
			if not ms:
				outputs = model(inputs)
				preds = torch.argmax(outputs, 1)
			if ms:
				for img in ms_img:
					img = img.float().cuda()
					outputs = model(img)
					outputs = F.interpolate(outputs, size=size,
						mode='bilinear', align_corners=True)
					scores += outputs
					if flip:
						img_f = torch.flip(img, dims=(3,))
						a = model(img_f)
						logits = a
						logits = torch.flip(logits, dims=(3,))	
						outputs = F.interpolate(logits, size=size,
						mode='bilinear', align_corners=True)
						scores += outputs
				preds = torch.argmax(scores, 1)

			confidence = torch.softmax(outputs, 1).max(1)[0]
			loss = criterion(outputs, labels)
			
			# Statistics
			bs = inputs.size(0) # current batch size
			loss = loss.item()
			loss_running.update(loss, bs)
			corrects = torch.sum(preds == labels.data)
			nvoid = int((labels==void).sum())
			acc = corrects.double()/(bs*res-nvoid)
			acc_running.update(acc, bs)

			# Calculate IoU scores of current batch
			iou.evaluateBatch(preds, labels)
			
			# Save visualizations
			if save:
				for i in range(inputs.size(0)):
					filename = os.path.splitext(os.path.basename(filepath[i]))[0]
					img = visim(inputs[i,:,:,:], mean, std)
					label = vislbl(labels[i,:,:], maskColors)
					pred = vislbl(preds[i,:,:], maskColors)
					cv2.imwrite(save_path + f'/visuals/{filename}-{save_suffix}.png',pred[:,:,::-1])
					cv2.imwrite(save_path + f'/visuals/{filename}-gt.png',label[:,:,::-1])
					# cv2.imwrite(save_path + f'/visuals/{filename}-original.png',img[:,:,::-1])


			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			
			# print progress info
			progress.display(epoch_step)
		
		miou, iou_summary, confMatrix = iou.outputScores(epoch=0)
		print(iou_summary)

	return acc_running.avg, loss_running.avg, miou, confMatrix, iou_summary