import torch
import torch.nn as nn
from torch import optim
import numpy as np
import imageio
from poisson_blend import blend

# helper function to convert tensor image to numpy array
def denorm(x):
  x = x.to('cpu')
  x = x.detach().numpy().squeeze()
  x = np.transpose(x, (1, 2, 0))
  x = x*np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5));
  x = x.clip(0, 1)
  return x

#context loss
def contextLoss(corrupt_imgs, gen_imgs, masks):
    """
    Args:
        corrupt_imgs: corrupted input images
        gen_imgs: generated images
        masks: weighted_mask or mask
    """
    a = gen_imgs*masks
    b = corrupt_imgs*masks
    L_context = torch.sum(abs(a-b))
    return L_context


def inPaint(trainloader, netG, netD, nNoise, args):
    device = args.device
    for idx, (f_name, corrupted_imgs, original_imgs, masks, weighted_masks) in enumerate(trainloader):
        #file name of image
        name = f_name
        
        #loading data onto device for calculation
        corrupted_imgs = corrupted_imgs.to(device)
        original_imgs = original_imgs.to(device)
        weighted_masks = weighted_masks.to(device)
        masks = masks.to(device)
        
        # encoding to be updated during backpropagation
        z = nn.Parameter(torch.randn((corrupted_imgs.size(0), nNoise, 1, 1), device="cuda"))
        optimizer = optim.Adam([z])
        
        # setting generator and discriminator to inference method
        netG.eval()
        netD.eval()
        
        # optimization loop
        print("Inpainting started...")
        for i in range(args.epochs):
            optimizer.zero_grad()
            generated_imgs = netG(z)
            d_out = netD(generated_imgs)
            L_c = contextLoss(corrupted_imgs, generated_imgs, weighted_masks)
            # L_p = prior_loss(d_out.view(-1, d_out.size(0)).squeeze(0), real_lbl)
            L_p = torch.sum(torch.log(1.-d_out))
            loss = L_c + args.lmbda*L_p
            loss.backward()
            optimizer.step()
        
        print('inpainting completed for batch: {}'.format(idx+1))
        
        # saving generated images
        for j in range(len(name)):
            #blended image
            blended_img = blend(denorm(corrupted_imgs[j]), denorm(generated_imgs[j]), (1.-masks[j]))
            path_b = args.save_path + 'blended_' + name[j]
            imageio.imsave(path_b, (blended_img))
           
            #generated image
            generated_img = denorm(generated_imgs[j])
            path_g = args.save_path + 'generated_' + name[j]
            imageio.imsave(path_g, (generated_img))
            print("regenerated image for image {} saved at {}".format(name[j], 
