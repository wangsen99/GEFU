import torch
from torch import nn
import argparse
import os
from torchvision import transforms
from utils.utils import *
from utils.resnet import resnet18
from utils.resnet_BYOL import ResNet_BYOL
from codan import CODaN

parser = argparse.ArgumentParser(description='Classification')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pt',
					help='location for pre-trained daytime model')
parser.add_argument('--gpu_id', default='0', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def validate(model, test_night_loader):
	model.eval()
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()
	top1 = AverageMeter()
	losses = AverageMeter()
	acc = []
	for loader in [test_night_loader]:
		losses.reset()
		top1.reset()
		for images, labels in loader:
			images = images.cuda()
			labels = labels.cuda()

			with torch.no_grad():
				outputs = model(images)
				loss = criterion(outputs,labels)

			prec1 = accuracy(outputs.data, labels)[0]
			top1.update(prec1.item(), images.size(0))
			losses.update(float(loss.detach().cpu()))
		acc.append(top1.avg)
		print(f"Accuracy: {top1.avg:.2f}\t Loss: {losses.avg:.4f}")
	return acc

def main():
	transforms_test = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
	test_night_dataset = CODaN(split='gefu_cls',transform=transforms_test)
	test_night_loader = torch.utils.data.DataLoader(test_night_dataset,
			num_workers=8,
			batch_size=args.batch_size,
			shuffle=False)
	
	state_dict = torch.load(args.checkpoint)
	resnet = resnet18(num_classes=10)
	if 'state_dict' in state_dict:
		state_dict = state_dict['state_dict']
	resnet.load_state_dict(state_dict, strict=False)
	model = ResNet_BYOL(num_classes=10, resnet=resnet)
	model.cuda()
 
	acc = validate(model, test_night_loader)
	print('accuracy', acc)

if __name__ == '__main__':
	main()
