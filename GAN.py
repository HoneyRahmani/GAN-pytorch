# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:38:59 2021

@author: asus
"""

#==============================================
from torchvision import datasets
import torchvision.transforms as transforms
#from torchvision.transforms.functional import Resize,
import os


path2data = "./data"
os.makedirs(path2data, exist_ok=True)

h,w = 64,64
mean = (0.5,0.5,0.5)
std = (0.5,0.5,0.5)

transform = transforms.Compose([
                            transforms.Resize((h,w)),
                            transforms.CenterCrop((h,w)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std) 
                            ])

train_ds = datasets.STL10(
                        path2data,split='train',
                        download = False,
                        transform = transform
                        )
print(len(train_ds))

###Get a sample tensor from the dataset
import torch

for x  , _ in train_ds:
    
    print(x.shape, torch.min(x), torch.max(x))
    break

for x, y in train_ds:
    print(x.shape, y)
    break
    
###Display a sample image

from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

plt.imshow(to_pil_image(0.5*x+0.5))

#=============================DataLoader
from torch.utils.data import DataLoader
batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
for x,y in train_dl:
    print(x.shape, y.shape)
    break

# ========== Define generator and discriminator
# === Generator

from torch import nn
import torch.nn.functional as F

class Generator (nn.Module):
    
    def __init__(self,params):
        super(Generator, self).__init__()
        
        nz = params["nz"]
        ngf = params["ngf"]
        noc = params["noc"]
        
        self.dcov1 = nn.ConvTranspose2d(nz, ngf*8, kernel_size=4, stride=1,
                                        padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(ngf*8)
        self.dcov2 = nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2,
                                        padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(ngf*4)
        self.dcov3 = nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2,
                                        padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(ngf*2)
        self.dcov4 = nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2,
                                        padding=1, bias=False)
        
        self.bn4 = nn.BatchNorm2d(ngf)
        self.dcov5 = nn.ConvTranspose2d(ngf, noc, kernel_size=4, stride=2,
                                        padding=1, bias=False)
        
    def forward(self, x):
        
        x = F.relu(self.bn1(self.dcov1(x)))
        x = F.relu(self.bn2(self.dcov2(x)))
        x = F.relu(self.bn3(self.dcov3(x)))
        x = F.relu(self.bn4(self.dcov4(x)))
        
        out = torch.tanh(self.dcov5(x))
        
        return out
    

params_gen = {
    "nz":100,
    "ngf":64,
    "noc":3
    }

model_gen = Generator(params_gen)
device = torch.device("cuda")
model_gen.to(device)
print(model_gen)

        
# =================================
        
with torch.no_grad():
    y = model_gen(torch.zeros(1, 100, 1, 1,device=device))
print(y.shape)
   

# === Discriminator

class Discriminator(nn.Module):
    
    def __init__(self,params):
        super(Discriminator, self).__init__()
        
        nic = params["nic"]
        ndf = params["ndf"]
        self.conv1 = nn.Conv2d(nic, ndf, kernel_size=4, stride=2, padding=1,bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(ndf*2)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(ndf*8)
        self.conv5 = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0,bias=False)
    def forward(self, x):
        x= F.leaky_relu(self.conv1(x), 0.2, True)
        x= F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True)
        x= F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True)
        x= F.leaky_relu(self.bn4(self.conv4(x)), 0.2, inplace=True)
        out = torch.sigmoid(self.conv5(x))
        return out.view(-1)
    

params_dis = {
    "nic": 3,
    "ndf":64
    }  

model_dis = Discriminator(params_dis)
model_dis.to(device)
print(model_dis)

with torch.no_grad():
    y= model_dis(torch.zeros(1,3,h,w,device=device))
print(y.shape)    


def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


model_gen.apply(initialize_weights);
model_dis.apply(initialize_weights);     
        
   
# ===== Defining the loss and optimizer
from torch import optim

loss_func = nn.BCELoss()
lr = 2e-4
beta1 =0.5
opt_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1,0.999))
opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(beta1,0.999))

# ===========================Training the models=========================

real_label = 1
fake_label = 0
nz = params_gen["nz"]
num_epoch = 50

loss_history = { "gen": [],
            "dis": []}

batch_count = 0
for epoch in range(num_epoch):
    for xb, yb in train_dl:
        ba_si= xb.size(0)
        #dis
        model_dis.zero_grad()
        xb = xb.to(device)
        yb = torch.full((ba_si,), real_label, device=device)
        out_dis = model_dis(xb)
        loss_r = loss_func(out_dis, yb)
        loss_r.backward()
        
        noise = torch.randn(ba_si, nz,1,1,device=device)
        out_gen = model_gen(noise)
        out_dis = model_dis(out_gen.detach())
        yb.fill_(fake_label)
        loss_f = loss_func(out_dis, yb)
        loss_f.backward()

        loss_dis = loss_r + loss_f
        
        opt_dis.step()
        
        #gen
        model_gen.zero_grad()
        yb.fill_(real_label)  
        out_dis = model_dis(out_gen)
        loss_gen = loss_func(out_dis, yb)
        loss_gen.backward()
        opt_gen.step()

        loss_history["gen"].append(loss_gen.item())
        loss_history["dis"].append(loss_dis.item())
        
        batch_count +=1
        
        if batch_count % 100 ==0:
            print(epoch, loss_gen.item(), loss_dis.item())
        
        
# ====Plot loss history

plt.figure(figsize=(10,5))
plt.title("Loss History")
plt.plot(loss_history["gen"],label="Gen.Loss")
plt.plot(loss_history["dis"], label="Dis.Loss")
plt.xlabel("Batch Count")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ==== Store the model

path2models = "./models/"
os.makedirs(path2models, exist_ok=True)
path2weight_gen=os.path.join(path2models,"weights_gen.pt")
path2weight_dis=os.path.join(path2models,"weights_dis.pt")

torch.save(model_gen.state_dict(), path2weight_gen)
torch.save(model_dis.state_dict(), path2weight_dis)
# ==== Deploy the generator

weights = torch.load(path2weight_gen)
model_gen.load_state_dict(weights)
model_gen.eval()

with torch.no_grad():
    fixed_noise = torch.randn(16,nz,1,1, device=device)
    img_fake = model_gen(fixed_noise).detach().cpu()
print(img_fake.shape)

#===== Display generated image
plt.figure(figsize=(10,10))
for ii in range(16):
    plt.subplot(4,4,ii+1)
    plt.imshow(to_pil_image(0.5*img_fake[ii]+0.5))
    plt.axis("off")
    

    

            

