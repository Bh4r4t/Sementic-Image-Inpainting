import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from trainGAN import loadModel
from dataset import PatchDataset
from inPainting import inPaint
import os

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2,
                        help="1 for single image")
    parser.add_argument("--gan_path", type=str, default="./saved_model/model.pth")
    parser.add_argument("--kernel_size", type=int, default=7,
                        help="kernel size to generate weighted mask")
    parser.add_argument("--mask_type", type=str, default='random',
                        help="random or center")
    parser.add_argument("--img_path", type=str, default="/images/imgs")
    parser.add_argument("--epochs", type=str, default=5000,
                        help="Number of iterations to update z")
    parser.add_argument("--lmbda", type=float, default = 0.003,
                        help="lambda(weight) of prior loss")
    parser.add_argument("--device", type=str, default = 'cpu')
    parser.add_argument("--save_path", type=str, default = './results/')
    
    args = parser.parse_args()
    return args


def main(args):
    img_size = 96
    nNoise = 100    # noise vector dimension (1 x nNoise x 1 x 1)
    nFeatures = 32
    nOut = 3
    
    #check for cuda compatible accelerator
    if torch.cuda.is_available() :
        args.device = 'cuda'
    
    # transforms for each input image
    tfms = transforms.Compose([transforms.Resize(img_size),
                           transforms.CenterCrop(img_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

    trainset = PatchDataset(args.img_path, img_size, transform = tfms, mask_type = args.mask_type, args = args)
    trainloader = DataLoader(trainset, shuffle=True, batch_size = args.batch_size)
    
    #check for saved model
    if not os.path.exists(args.gan_path) :
        print("Please load the saved GAN model or train a GAN and then try")
        exit()
    else:
        netG, netD = loadModel(args.gan_path, nNoise, nFeatures, nOut)
        netG.to(args.device)
        netD.to(args.device)
        
    # check for existence of result directory
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        print("result directory created")
    
    # inpainting process
    inPaint(trainloader, netG, netD, nNoise, args)

if __name__ == '__main__':
    args = get_arguments()
    main(args)
