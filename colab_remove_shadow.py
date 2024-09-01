!pip install memory_profiler
!pip install gradio
!pip install torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image, make_grid

class ResidualBlock(nn.Module):
    """Some Information about ResidualBlock"""
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256,256,kernel_size=3,stride=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256,256,kernel_size=3,stride=1),
            nn.InstanceNorm2d(256)
        )

    def forward(self, x):
        x = x + self.block(x)
        return x



class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self,img_channel,res_block):
        super(Generator, self).__init__()
        self.encode_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channel,64,kernel_size=7,stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,kernel_size=3,stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,256,kernel_size=3,stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        res_blocks = [ResidualBlock() for _ in range(res_block)]
        self.res_block = nn.Sequential(
            *res_blocks
        )
        self.decode_block = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64,img_channel,kernel_size=7,stride=1),
            nn.Tanh()
        )

    
    def forward(self, x):
        x = self.encode_block(x)
        x = self.res_block(x)
        x = self.decode_block(x)
        return x


class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self,img_channel):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(img_channel,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256,512,kernel_size=4,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1)
        )

    def forward(self, x):
        x = self.block(x)
        return x



import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/model.py')
# from model import Generator,Discriminator,init_weights

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from memory_profiler import profile
import gradio as gr
import datetime


def remove_shadow(image_np): #image:ndarray
    with torch.no_grad():
        # 元画像保存
        current_time = datetime.datetime.now()
        str_time = str(current_time.year)+'.'+str(current_time.month)+'.'+str(current_time.day)+'_'+str(current_time.hour)+'.'+str(current_time.minute)+'.'+str(current_time.second)
        pil_image = Image.fromarray(image_np)
        pil_image.save(save_path+"/"+ str_time+"_origin.png")

        image_tensor = torch.tensor(image_np)
        img_tensor = np.transpose(image_tensor, (2,0,1)) # 形状を変換する
        real_img_torch = img_tensor.to(device)
        real_img_torch_f = real_img_torch.to(torch.float32)
        # A：影あり、B：影なし
        G_A2B.load_state_dict(torch.load("/content/drive/My Drive/Colab Notebooks/models/199.pth"))
        trans_img_torch = G_A2B(real_img_torch_f)
        # 影取り後画像保存
        save_image(trans_img_torch, save_path+"/"+ str_time+"_convert.png", nrow=2, normalize=True)
        normalized_image = (trans_img_torch - trans_img_torch.min()) / (trans_img_torch.max() - trans_img_torch.min())
        save_image(normalized_image, save_path+"/B_test.png", nrow=2, normalize=False)
        trans_img_np = normalized_image.to('cpu').detach().numpy()
        data_transposed = np.transpose(trans_img_np, (1, 2, 0)) # 形状を変換する

    return data_transposed


if __name__ == "__main__":
    input_size = (256, 256)
    output_size = (256, 256)
    image_size = 256
    epoch = 200
    save_path = "/content/drive/My Drive/Colab Notebooks/result"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #set GPU or CPU
    gpu = 1
    if gpu >= 0 and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    #set depth of resnet
    if image_size == 128:
        res_block=6
    else:
        res_block=9
    #set models
    G_A2B = Generator(3,res_block).to(device)
    G_B2A = Generator(3,res_block).to(device)
    D_A = Discriminator(3).to(device)
    D_B = Discriminator(3).to(device)

    app = gr.Interface(remove_shadow, inputs="image", outputs="image", input_size=input_size, output_size=output_size, css="/content/drive/My Drive/Colab Notebooks/Blocks-005a10ea.css")
    # app.launch()
    app.launch(auth=("bear", "ltuG9OqDMBjYUtg"),debug=True,share=True)








