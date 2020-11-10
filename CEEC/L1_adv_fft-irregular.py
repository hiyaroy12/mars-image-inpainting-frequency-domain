"""
L1+adv+FFT: 
1.stage 1 loads random bbox pretrained models from "../DnCNN-PyTorch/logs" .. e.g. regular_celeba_net.pth
2.stage 2 trains the network for inpainting

CUDA_VISIBLE_DEVICES=6 python CEEC/L1_adv_fft-irregular.py --dataset hirise --use_irregular 1

"""

import os, sys
import numpy as np
import math, PIL
import argparse
from PIL import Image
sys.path.append( '../mars_frequency_inpainting' )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from CEEC.models import InpaintingModel, DnCNN
from CEEC.config import Config
from CEEC.dataloader import get_data_loader
from CEEC.metrics import PSNR

from freq_domain.utils import my_transform, read_file, MyCelebA, MyHirise
from freq_domain.utils import weights_init_kaiming, batch_PSNR, data_augmentation
# create_dir, create_mask, stitch_images, imshow, imsave, Progbar

from freq_domain.freq_utils import get_gray_images_back, get_gray_fft_images, get_gray_fft_images_irregular
# from freq_domain.freq_utils import read_file, product_mask, make_masked, random_bbox, get_color_images_back, get_color_fft_images, get_color_fft_images_regular, get_color_fft_images_irregular
from tensorboardX import SummaryWriter

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=35, help="size of the batches")
parser.add_argument("--dataset", type=str, default="hirise", help="name of the dataset")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
# parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--stage1_outf", type=str, default="./logs", help='path of log files')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--use_irregular", type=int, default=0, help='When irregular mask is used')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
opt = parser.parse_args()

config = Config('./CEEC/config_la_adv_fft.yml')
print(opt)

from datetime import datetime
now = datetime.now() # current date and time
datetime_f = "_".join(str(now).split())

use_irregular = bool(opt.use_irregular)
prefix = ('irregular_' if use_irregular else '')

os.makedirs("L1_adv_fft_results/{}{}_images".format(prefix, opt.dataset), exist_ok=True)
use_cuda = not opt.no_cuda and torch.cuda.is_available()
torch.manual_seed(1234)
device = torch.device("cuda" if use_cuda else "cpu")

# PSNR metric
psnr_compute = PSNR(255.0).to(config.DEVICE)

# ----------
#  Load stage1
# ----------

def load_stage1_model():
#     net_stage1 = DnCNN(channels=12, out_ch=6, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    net_stage1 = DnCNN(channels=4, out_ch=2, num_of_layers=opt.num_of_layers)
    net_stage1 = nn.DataParallel(net_stage1, device_ids=device_ids).cuda()
    if use_irregular:
        PATH = '{}/irregular_{}_net.pth'.format(opt.stage1_outf, opt.dataset)
    print('Loading from ', PATH)
    ckpt = torch.load(PATH, map_location='cpu')
    net_stage1.load_state_dict(ckpt)
    return net_stage1

net_stage1 = load_stage1_model().to(device)
# ----------
#  Dataloader
# ----------
loader_train, test_loader = get_data_loader(opt)
# ----------
#  Call stage2 for training
# ----------
model = InpaintingModel(config, gen_in_channel=3, disc_in_channel=1, gen_out_channels=1).to(device)
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(loader_train):     
        
        ############ Mask the images and compute the FFTs ##########################################################
        if use_irregular:
            outputs_irregular = get_gray_fft_images_irregular(imgs.numpy(), True)
            x_masked, x_fft, x_masked_fft, lims_list, idx_list, idx_list_m, all_masks, mask_fft = outputs_irregular

        masked_imgs = torch.from_numpy(x_masked).type(Tensor) 
        all_masks = np.expand_dims(all_masks, axis=1)
        masks = torch.from_numpy(all_masks)
#         masks = np.transpose(masks,(0,3,1,2))
        masks = masks.type(Tensor)                           
        imgs = imgs.type(Tensor) 

    
        img_train = torch.from_numpy(x_fft).type(Tensor)                                #changed (x_fft~(b, 2, 256, 256))
        imgn_train = torch.from_numpy(x_masked_fft).type(Tensor)                        #changed (x_masked_fft~(b, 2, 256, 256))
        img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda()) #   (img_train,imgn_train~torch.Size([b, 2, 256, 256]))

        mask_fft = torch.from_numpy(mask_fft).type(torch.FloatTensor)                   #added   (mask_fft~torch.Size([b, 2, 256, 256])
        mask_train_in = Variable(mask_fft.cuda())                                       #added   (mask_train_in~torch.Size([b, 2, 256, 256])
        
        imgn_train_cat = torch.cat((mask_train_in, imgn_train), axis=1)  
        # imgn_train_cat~torch.Size([40, 4, 256, 256])
        out_train = torch.clamp(net_stage1(imgn_train_cat), 0., 1.)             #(out_train~torch.Size([b, 2, 256, 256]))      
        
#         import ipdb; ipdb.set_trace()
        img_back = get_gray_images_back(img_train.cpu().numpy(), lims_list, idx_list)             #img_back~(b, 1, 256, 256)
        img_back_masked = get_gray_images_back(imgn_train.cpu().numpy(), lims_list, idx_list_m)   #img_back_masked~(b, 1, 256, 256)
        img_back_recon = get_gray_images_back(out_train.detach().cpu().numpy(), lims_list, idx_list) #img_back_recon~(b, 1, 256, 256)
        img_back_recon = torch.clamp(torch.from_numpy(img_back_recon), -1., 1.).type(Tensor)        

        masked_imgs_display = masked_imgs.clone()                                  #masked_imgs_display: torch.Size([b, 1, 256, 256])
        masked_imgs = torch.cat((masked_imgs, img_back_recon), axis=1)         
        # masked_imgs=masked_imgs+ifft of 1st_stage torch.Size([b, 2, 256, 256])
        
        i_outputs, i_gen_loss, i_dis_loss, i_logs = model.process(imgs, masked_imgs, masks)
        #imgs.shape: torch.Size([b, 1, 256, 256]) 
        #masked_imgs.shape: torch.Size([b, 2, 256, 256])
        #masks.shape: torch.Size([b, 1, 256, 256])
        #i_outputs.shape: torch.Size([40, 1, 256, 256])
        outputs_merged = (i_outputs * (1 - masks)) + (imgs * masks)
        #outputs_merged.shape: torch.Size([40, 1, 256, 256])

        ########################################################################################################
        # metrics
        psnr = psnr_compute(postprocess((imgs+1.)/2.), postprocess((outputs_merged+1.)/2.))
        mae = (torch.sum(torch.abs(imgs - outputs_merged)) / torch.sum(imgs)).float()
        
        i_logs.append(('psnr', psnr.item()))
        i_logs.append(('mae', mae.item()))
        
        print("[Epoch %d/%d] [Batch %d/%d]"% (epoch, opt.n_epochs, i, len(loader_train)))
        for log in i_logs:
            print(log[0]+' : '+str(log[1]))
            
        # backward
        model.backward(i_gen_loss, i_dis_loss)
        iteration = model.iteration
              
        # Generate sample at sample interval
        batches_done = epoch * len(loader_train) + i
        if batches_done % opt.sample_interval == 0:
            sample = torch.cat((masked_imgs_display.data, outputs_merged.data, imgs.data), -2)
            save_image(sample, "L1_adv_fft_results/{}{}_images/%d.png".format(prefix, opt.dataset) % batches_done, nrow=8, normalize=True)
    
    torch.save(model.generator.state_dict(),"L1_adv_fft_results/{}{}_generator.h5f".format(prefix, opt.dataset))
    torch.save(model.discriminator.state_dict(),"L1_adv_fft_results/{}{}_discriminator.h5f".format(prefix, opt.dataset))
     
    
    
