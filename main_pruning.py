from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import pdb
import os
import numpy as np
import torch.nn.functional as F
import copy

from Util import *
import resnet

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--step_ft', type=int, default=300)
parser.add_argument('--ft_lr', type=float, default=1e-3)
parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--workers', type= int, default= 4)
parser.add_argument('--arch', type=str, default='resnet32')
parser.add_argument('--model', type=str, default=None)

def main():
	global args, iters
	global file
	args = parser.parse_args()

	args.gpu = [int(i) for i in args.gpu.split(',')]	
	torch.cuda.set_device(args.gpu[0] if args.gpu else None)	
	torch.backends.cudnn.benchmark = True
	L_cls_f = nn.CrossEntropyLoss().cuda()

	## Dataset Loading 
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(root='./cifar10', train=True, transform=transforms.Compose([
			transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
			transforms.ToTensor(), normalize]), download=True),
		batch_size=args.batch_size, shuffle=True,
		num_workers= args.workers, pin_memory=True)

	val_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(root='./cifar10', train=False, transform=transforms.Compose([
			transforms.ToTensor(),normalize])),
		batch_size=128, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	## Model Initialize and Loading
	model = resnet.resnet32()
	args.checkpoint = 'models/model_32.th'

	model = nn.DataParallel(model, device_ids=args.gpu).cuda()	
	checkpoint = torch.load(args.checkpoint,map_location='cuda:0')
	model.load_state_dict(checkpoint['state_dict'])

	model.eval()
	original_model = copy.deepcopy(model)

	loss, init_acc = validate(val_loader, model, L_cls_f, '')
	print('\nOriginal performance. Acc: {:2.2f}%'.format(init_acc))

	num_params = get_params(model)
	num_flops = get_flops(model)

	## 1. Initialization process
	layer_names = get_layer_names(model)
	expected_flops(copy.deepcopy(model), layer_names[1:], num_params, num_flops, args.ratio)
	add_1x1_convs(model, layer_names[1:])

	print('== 1. Initialization fine-tuning stage. ')
	model_opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
	for epochs in range(20):		
		fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt, False)
		loss, acc = validate(val_loader, model, L_cls_f, '* ')

		print("[Init {:02d}] Loss: {:.3f}. Acc: {:2.2f}%. || Param: {:2.2f}%  Flop: {:2.2f}%".format(epochs+1, loss, acc, \
			get_params(model)/num_params*100, get_flops(model)/num_flops*100))	

	## 2. Pruning process, from top to bottom
	print('\n== 2. Pruning stage. ')
	for i in range(1,len(layer_names)):
		index = len(layer_names)-i
		model = pruning_output_channel(model, original_model, layer_names[index], train_loader, val_loader, L_cls_f)		
		model = pruning_input_channel(model, original_model, layer_names[index], train_loader, val_loader, L_cls_f)

		loss, acc = validate(val_loader, model, L_cls_f, '* ')
		print("[Pruning {:02d}]. Loss: {:.3f}. Acc: {:2.2f}%. || Param: {:2.2f}%  Flop: {:2.2f}%".format(index, loss, \
			acc, get_params(model)/num_params*100, get_flops(model)/num_flops*100))
	
	## 3. Final Fine-tuning stage
	print('\n==3. Final fine-tuning stage after pruning.')
	best_acc = 0
	model_opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

	for epochs in range(args.step_ft):
		adjust_learning_rate(model_opt, epochs, args.step_ft)
		fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)
		loss, acc = validate(val_loader, model, L_cls_f, '* ')
		if acc > best_acc: best_acc = acc

		print("[Fine-tune {:03d}] Loss: {:.3f}. Acc: {:2.2f}%. ||\
		 Param: {:2.2f}%  Flop: {:2.2f}%  Best: {:2.2f}%".format(epochs+1, loss, acc, \
			get_params(model)/num_params*100, get_flops(model)/num_flops*100, best_acc))

	print("\n[Final] Baseline: {:2.2f}%. After Pruning: {:2.2f}%. ||\
	 Diff: {:2.2f}%  Param: {:2.2f}%  Flop: {:2.2f}%".format(\
	 	init_acc, best_acc, init_acc - best_acc, get_params(model)\
	 	/num_params*100, get_flops(model)/num_flops*100))

def pruning_output_channel(model, original_model, layer_name, train_loader, val_loader, L_cls_f):

	global args

	#Calculate the number of channels to prune
	layer_name = layer_name.split('.')[:-1]
	layer = model
	for i in range(len(layer_name)-1):
		layer = layer._modules[layer_name[i]]
	conv3x3 = layer._modules[layer_name[-1]][1]

	pruning_ratio = args.ratio
	pruning_num = int(round(conv3x3.out_channels * pruning_ratio))
	pruning_num = min(conv3x3.out_channels-1, pruning_num)

	for xx in range(pruning_num):

		layer = model
		for i in range(len(layer_name)-1):
			layer = layer._modules[layer_name[i]]
		conv3x3 = layer._modules[layer_name[-1]][1]
		conv1x1 = layer._modules[layer_name[-1]][2]		
		mid_channel = conv3x3.out_channels

		new_conv3x3 = nn.Conv2d(in_channels = conv3x3.in_channels, out_channels = conv3x3.out_channels - 1, kernel_size = conv3x3.kernel_size, \
				stride = conv3x3.stride, padding = conv3x3.padding, bias = conv3x3.bias).cuda()
		new_conv1x1 = nn.Conv2d(in_channels = conv1x1.in_channels - 1, out_channels = conv1x1.out_channels, kernel_size = (1,1), \
				stride = conv1x1.stride, padding = (0,0), bias = conv3x3.bias).cuda()

		##LRF calculation
		conv3x3_weights = conv3x3.weight.data.view(conv3x3.weight.data.shape[0], -1)
		error_list = torch.zeros(mid_channel)
		lambda_list = torch.zeros((mid_channel -1 , mid_channel)).cuda()

		for ch in range(mid_channel):

			weights = torch.cat((conv3x3_weights[:ch, :], conv3x3_weights[ch+1:,:]), 0)
			A_mat = torch.mm(weights, weights.transpose(0,1))
			B_mat = torch.mm(weights, conv3x3_weights[ch, :].unsqueeze(1))
			lamb = torch.mm(A_mat.inverse(), B_mat)

			epsilon = conv3x3_weights[ch, :] - torch.mm(weights.transpose(0,1), lamb).squeeze()
			error_list[ch] = epsilon.norm().item() * conv1x1.weight.data[:,ch,:,:].norm().item()
			lambda_list[:, ch] = lamb.squeeze()

		min_id = torch.argmin(error_list) # Channel index with the lowest approximation error
		lambda_id = lambda_list[:, min_id]

		## Copy the weight values of original convolution to new convolution
		## except the channel with the lowest approximation error
		new_conv3x3.weight.data[:min_id,:,:,:] = conv3x3.weight.data[:min_id,:,:,:]
		new_conv3x3.weight.data[min_id:,:,:,:] = conv3x3.weight.data[min_id+1:,:,:,:]
		new_conv1x1.weight.data[:,:min_id,:,:] = conv1x1.weight.data[:,:min_id,:,:]
		new_conv1x1.weight.data[:,min_id:,:,:] = conv1x1.weight.data[:,min_id+1:,:,:]	

		## Weights Compensation
		conv_shape = conv1x1.weight.data.shape
		compen_weight = conv1x1.weight.data[:,[min_id],:,:].repeat(1, mid_channel-1, 1, 1)
		compen_weight = compen_weight * lambda_id.cuda().view(1,-1,1,1).repeat(conv_shape[0], 1, conv_shape[2], conv_shape[3])
		new_conv1x1.weight.data = new_conv1x1.weight.data + compen_weight

		layer._modules[layer_name[-1]][1] = new_conv3x3
		layer._modules[layer_name[-1]][2] = new_conv1x1

	#One epoch of fine-tuning
	model_opt = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
	fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)

	return model


def pruning_input_channel(model, original_model, layer_name, train_loader, val_loader, L_cls_f):

	global args

	layer_name = layer_name.split('.')[:-1]
	layer = model
	for i in range(len(layer_name)-1):
		layer = layer._modules[layer_name[i]]
	conv3x3 = layer._modules[layer_name[-1]][1]

	#Calculate the number of channels to prune
	pruning_ratio = args.ratio
	pruning_num = int(round(conv3x3.in_channels * pruning_ratio))
	pruning_num = min(conv3x3.in_channels-1, pruning_num)

	for xx in range(pruning_num):

		layer = model
		for i in range(len(layer_name)-1):
			layer = layer._modules[layer_name[i]]
		conv1x1 = layer._modules[layer_name[-1]][0]
		conv3x3 = layer._modules[layer_name[-1]][1]		
		mid_channel = conv1x1.out_channels

		new_conv1x1 = nn.Conv2d(in_channels = conv1x1.in_channels, out_channels = conv1x1.out_channels - 1, kernel_size = (1,1), \
				stride = (1,1), padding = (0,0), bias = conv3x3.bias).cuda()
		new_conv3x3 = nn.Conv2d(in_channels = conv3x3.in_channels - 1, out_channels = conv3x3.out_channels, kernel_size = conv3x3.kernel_size, \
				stride = conv3x3.stride, padding = conv3x3.padding, bias = conv3x3.bias).cuda()

		##LRF calculation
		conv3x3_weights = conv3x3.weight.data.transpose(0,1).contiguous().view(conv3x3.weight.data.shape[1], -1)
		error_list = torch.zeros(mid_channel)
		lambda_list = torch.zeros((mid_channel -1 , mid_channel)).cuda()

		for ch in range(mid_channel):

			weights = torch.cat((conv3x3_weights[:ch, :], conv3x3_weights[ch+1:,:]), 0)
			A_mat = torch.mm(weights, weights.transpose(0,1))
			B_mat = torch.mm(weights, conv3x3_weights[ch, :].unsqueeze(1))
			lamb = torch.mm(A_mat.inverse(), B_mat)

			epsilon = conv3x3_weights[ch, :] - torch.mm(weights.transpose(0,1), lamb).squeeze()
			error_list[ch] = epsilon.norm().item() * conv1x1.weight.data[ch,:,:,:].norm().item()
			lambda_list[:, ch] = lamb.squeeze()

		min_id = torch.argmin(error_list) # Channel index with the lowest approximation error
		lambda_id = lambda_list[:, min_id]

		## Copy the weight values of original convolution to new convolution
		## except the channel with the lowest approximation error
		new_conv1x1.weight.data[:min_id,:,:,:] = conv1x1.weight.data[:min_id,:,:,:]
		new_conv1x1.weight.data[min_id:,:,:,:] = conv1x1.weight.data[min_id+1:,:,:,:]
		new_conv3x3.weight.data[:,:min_id,:,:] = conv3x3.weight.data[:,:min_id,:,:]
		new_conv3x3.weight.data[:,min_id:,:,:] = conv3x3.weight.data[:,min_id+1:,:,:]	

		## Weights Compensation
		conv_shape = conv1x1.weight.data.shape
		compen_weight = conv1x1.weight.data[[min_id],:,:,:].repeat(mid_channel-1, 1, 1, 1)
		compen_weight = compen_weight * lambda_id.cuda().view(-1,1,1,1).repeat(1, conv_shape[1], conv_shape[2], conv_shape[3])
		new_conv1x1.weight.data = new_conv1x1.weight.data + compen_weight

		layer._modules[layer_name[-1]][0] = new_conv1x1
		layer._modules[layer_name[-1]][1] = new_conv3x3

	#One epoch of fine-tuning
	model_opt = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
	fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)

	return model

def distillation_loss(y_logit, t_logit, T=2):
   return F.kl_div(F.log_softmax(y_logit/T, 1), F.softmax(t_logit/T, 1), reduction='sum')/y_logit.size(0)

def fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt, use_distill=True):

	global args	
	model.train()

	for i, (input, target) in enumerate(train_loader):

		target = target.cuda(non_blocking=True)	
		model_opt.zero_grad()
		z = model(input)
		z_ori = original_model(input)
		L = L_cls_f(z, target)
		if use_distill:
			L += distillation_loss(z, z_ori)
		L.backward()
		model_opt.step()	
	model.eval()

def adjust_learning_rate(optimizer, epoch, total):
    lr = 0.01
    if epoch > total/2: lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(val_loader, model, L_cls_f, prefix='', print=False):
	global args

	loss=0
	model.eval()
	
	total=0
	correct=0

	with torch.no_grad():
		for i, (input, target) in enumerate(val_loader):
			target = target.cuda(non_blocking=True)
			
			z = model(input)
			L_cls = L_cls_f(z, target)
			loss += L_cls.item()

			_, predicted = torch.max(z.data, 1)
			total += input.size(0)
			correct += (predicted == target).sum().item()

	if print: print('== {} Loss : {:.5f}. Acc : {:2.2f}%'.format(prefix, loss/len(val_loader), correct/total*100))

	return loss/len(val_loader), correct/total*100

if __name__ == '__main__':
	main()