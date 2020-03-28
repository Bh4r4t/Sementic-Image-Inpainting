# Semantic Image Inpainting with Deep Generative Models
PyTorch implementation of research paper [Semantic Image Inpainting with Deep Generative Models by R.A. Yeh et al.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf)


**saved_model** contains pretrained GAN parameter dictionary
required for inference during `inpainting`  
**images** contains image files for inpainting
**img** contains image files for GAN training
**trainGAN.py** script to train and save the state dictionary of GAN
**main.py** entry point of script for inpainting 


### Install all dependencies
```
pip install -r requirements.txt
```

### To train GAN
1. This script assumes that the path to training images provided has a subfolder, and all images are inside that subfolder.
To train GAN
```
$ py trainGAN.py
```
3. Current Saved model is trained on [CelebA dataset](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8). Saved model is present in `saved_model` folder.

### To inpaint an image
1. Script takes clean image(original image) as input and generates a patchy image out of that and tries to recover that patchy image. Meanwhile the original image is not used.
```
$ py main.py
```
After the inpainting task is completed the inpainted image is saved at the desired location

### Results
