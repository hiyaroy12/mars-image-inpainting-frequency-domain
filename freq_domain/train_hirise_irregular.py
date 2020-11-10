#CUDA_VISIBLE_DEVICES=6 python freq_domain/train_hirise_irregular.py --dataset hirise --use_irregular 1

import os
import argparse
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *
from freq_utils import get_gray_images_back, get_gray_fft_images, get_gray_fft_images_irregular

import argparse
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets.celeba import CelebA

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--batchSize", type=int, default=40, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--dataset", type=str, default="hirise", help='The dataset being used')
parser.add_argument("--use_irregular", type=int, default=0, help='The dataset being used')

opt = parser.parse_args()
print(opt)
from datetime import datetime
now = datetime.now() # current date and time
datetime_f = "_".join(str(now).split())

use_irregular = bool(opt.use_irregular)
prefix = ('irregular_' if use_irregular else '')

def main():
    # Load dataset
    print('Loading dataset ...\n')
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1234)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        
    if opt.dataset=='hirise':
        dataset_train = MyHirise('../data', split="train", download=False,
                                   transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.5,), 
                                                                                      (0.5,))]))

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True)

        dataset_val = MyHirise('../data', split="valid", 
                                    transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.5,),
                                                                                      (0.5,))]))
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=True)


    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
#     net = DnCNN(channels=2, num_of_layers=opt.num_of_layers)
    net = DnCNN(channels=4, out_ch=2, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False) # ToDo: Add weighted MSE loss
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
#     logdir = opt.outf + '/{}_{}'.format(opt.dataset, datetime_f)
    logdir = opt.outf + '/{}{}_{}'.format(prefix, opt.dataset, datetime_f)
    os.makedirs(logdir, exist_ok=True)
    
#     writer = SummaryWriter(opt.outf + '/tb_{}_{}'.format(opt.dataset, datetime_f))
    writer = SummaryWriter(opt.outf + '/{}tb_{}_{}'.format(prefix, opt.dataset, datetime_f))
    step = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data[0]
            
            if use_irregular:
                x_masked, x_fft, x_masked_fft, lims_list, idx_list, idx_list_m, all_masks, mask_fft = get_gray_fft_images_irregular(img_train.numpy(), True)

            #for Olivetti
#             x_masked, x_fft, x_masked_fft, lims_list, idx_list, idx_list_m = get_gray_fft_images(img_train.numpy(), dx=12, 
#                                                                                                  half=True if opt.dataset=='olivetti' else False)
            
           
            # img_back = get_gray_images_back(x_fft, lims_list, idx_list)
            # img_back_masked = get_gray_images_back(x_masked_fft, lims_list, idx_list_m)

            img_train = torch.from_numpy(x_fft).type(torch.FloatTensor) #np.zeros((N, 2, h, w))
            imgn_train = torch.from_numpy(x_masked_fft).type(torch.FloatTensor) #np.zeros((N, 2, h, w))
            mask_fft = torch.from_numpy(mask_fft).type(torch.FloatTensor) #np.zeros((N, 2, h, w))
            
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            mask_train_in = Variable(mask_fft.cuda())
            imgn_train_cat = torch.cat((mask_train_in, imgn_train), axis=1) #np.zeros((N, 4, h, w))
            
#             import ipdb; ipdb.set_trace()

            out_train = model(imgn_train_cat)
            loss = criterion(out_train, img_train) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            # out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            out_train = torch.clamp(model(imgn_train_cat), 0., 1.)
            
            img_back = get_gray_images_back(img_train.cpu().numpy(), lims_list, idx_list)
            img_back_masked = get_gray_images_back(imgn_train.cpu().numpy(), lims_list, idx_list_m)
            img_back_recon = get_gray_images_back(out_train.detach().cpu().numpy(), lims_list, idx_list)
            
            img_back = (torch.from_numpy(img_back) + 1)/2
            img_back_masked = (torch.from_numpy(img_back_masked) + 1)/2
            img_back_recon = (torch.from_numpy(img_back_recon) + 1)/2
            
            psnr_train = batch_PSNR(img_back, img_back_masked, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            

            ## the end of each epoch
            model.eval()
            if step % 50 == 0:
                print('Saving images...')
                Img = utils.make_grid(img_back.data, nrow=8, normalize=True, scale_each=True)
                Imgn = utils.make_grid(img_back_masked.data, nrow=8, normalize=True, scale_each=True)
                Irecon = utils.make_grid(img_back_recon.data, nrow=8, normalize=True, scale_each=True)
                writer.add_image('clean image', Img, epoch)
                writer.add_image('noisy image', Imgn, epoch)
                writer.add_image('reconstructed image', Irecon, epoch)

                utils.save_image(img_back.data, logdir + '/clean_image_{}.png'.format(step//50))
                utils.save_image(img_back_masked.data, logdir + '/noisy_image_{}.png'.format(step//50))
                utils.save_image(img_back_recon.data, logdir + '/reconstructed_image_{}.png'.format(step//50))
                
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, '{}{}_net.pth'.format(prefix, opt.dataset)))

if __name__ == "__main__":
    main()
