import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import os

CHECK_PATH = './saved_model/'

def getArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_path", type=str, default="./saved_model/")
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--real_lbl", type=float, default=0.9)
    parser.add_argument("--fake_lbl", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    # here this code accepts the image path such that, images are stored
    # inside another folder at path train_imgs
    # for example in default case images are stored in another folder inside images folder
    parser.add_argument("--train_imgs", type=str, default='C:/Users/Ladre/.spyder-py3/sementicImgInp/images/',
                        help="path to the directory of the training images")
    return parser.parse_args()

    
def saveModel(netG, optimizerG, netD, optimizerD, filepath):
    """
    Args:
        netG: generator network
        optimizerG: generator optimizer
        netD: discriminator network
        optimizerD: discriminator optimizer
        filepath: (optional) model parameters are saved at "filepath"
    """
    if not os.path.isdir(filepath) :
        os.mkdir(filepath) 
    filepath = filepath+'model.pth'
        
    torch.save({
            'G_state_dict': netG.state_dict(),
            'G_optimizer_state_dict':optimizerG.state_dict(),
            'D_state_dict': netD.state_dict(),
            'D_optimizer_state_dict':optimizerD.state_dict(),
            }, filepath)
    print("Model saved at {}".format(filepath))


def loadModel(filepath, nNoise, nFeatures, nOut):
    """
    Args:
        filepath: model
        nNoise: (100) size of latent space vector
        nFeatures: number of features for net
        nOut: number of output channel from generator
    """
    state_dict = torch.load(filepath)
    netG = Generator(nNoise, nFeatures, nOut)
    netD = Discriminator(nFeatures)
    netG.load_state_dict(state_dict['G_state_dict'])
    netD.load_state_dict(state_dict['D_state_dict'])
    
    print("Model loaded from {}".format(filepath))
    return netG, netD


def loadData(args):
    img_size= 96
    tfms = transforms.Compose([transforms.Resize(img_size),
                           transforms.CenterCrop(img_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

    trainset = datasets.ImageFolder(args.train_imgs, transform = tfms)
    trainloader = DataLoader(trainset, shuffle=True, batch_size = args.batch_size)
    return trainloader


def train(args):
    nNoise = 100
    nFeatures = 32
    nOut = 3
    
    print("loading images...")
    trainloader = loadData(args)
    
    #check for cuda compatible accelerator
    if torch.cuda.is_available() :
        device = 'cuda'
    else:
        device = 'cpu' 
    print('device: {}'.format(device))
    
    # generator and discriminator intialization
    netG = Generator(nNoise, nFeatures, nOut).to(device)
    netD = Discriminator(nFeatures).to(device)
    
    # optimizer
    optimizerD = optim.Adam(netD.parameters(), lr= args.lr, betas = (args.momentum, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr= args.lr, betas = (args.momentum, 0.999))
    
    # loss functions
    criterion = nn.BCELoss()
    
    batch_len = len(trainloader)
    
    #Training Loop
    print('GAN training started...')
    for e in range(args.epochs):
        for idx, (imgs,_) in enumerate(trainloader):
            ##########################
            # Discriminator training #
            ##########################
            netD.zero_grad()
            bs = imgs.size(0)
            # real image training
            images = imgs.to(device)
            real_lbl = torch.full((bs, 1), args.real_lbl).to(device)
            real_d_out = netD(images)
            real_loss = criterion(real_d_out, real_lbl)
            real_loss.backward()
            real_loss = real_loss.mean()
        
            # fake image training
            noise = torch.empty(bs, nNoise, 1, 1).normal_(0, 1.0).to(device)
            fake_imgs = netG(noise)
            fake_d_out = netD(fake_imgs.detach())
            fake_lbl = torch.full((bs, 1), args.fake_lbl).to(device)
            fake_loss = criterion(fake_d_out, fake_lbl)
            fake_loss.backward()
            fake_loss = fake_loss.mean()
        
            d_loss = real_loss + fake_loss
            optimizerD.step()
        
            ##########################
            #   Generator training   #
            ##########################
            netG.zero_grad()
        
            g_d = netD(fake_imgs)
            gen_loss = criterion(g_d, real_lbl)
            gen_loss.backward()
            optimizerG.step()
        
        print('Epoch: {:2}/{} | Batch = {:4}/{:4} | fake_loss: {:.5f} | real_loss: {:.5f} | d_loss: {:.5f} | g_loss: {:.5f}'.format(e+1,args.epochs, idx+1, batch_len, fake_loss.item(), real_loss.item(), d_loss.item(), gen_loss.item()))
             
    saveModel(netG, optimizerG, netD, optimizerD, args.gan_path)
    
if __name__ == '__main__':
    args = getArguments()
    train(args)
              
