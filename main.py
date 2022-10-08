import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt

def denormalization(x,channels=None,w=None,h=None,resize=False):
    x=0.5*(x+1)
    x=x.clamp(0,1)
    if resize:
        if channels is None or h is None or w is None:
            print("Please input channels,h,w")
        x=x.view(x.size(0),channels,h,w)
    return x

def show(img):
    if torch.cuda.is_available():
        img=img.cpu()
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))

#device selection 
GPU=True 
dev_id=0
if GPU:
    device=torch.device("cuda :"+str(dev_id)if torch.cuda.is_available() else "cpu")
else:
    device=torch.device("cpu")
print("Device:",device)

#reproduce
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic=True
torch.manual_seed(0)

#data loading

batch_size=128
if not os.path.exists('./CW/CAE'):
    os.makedirs('./CW/CAE')
if not os.path.exists('./CW/DCGAN'):
    os.makedirs('./CW/DCGAN')

NUM_TRAIN=49000
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])



transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

data_dir = './datasets'
cifar10_train = datasets.CIFAR10(data_dir, train=True, download=True,
                             transform=transform_train)
cifar10_val = datasets.CIFAR10(data_dir, train=True, download=True,
                           transform=transform_test)
cifar10_test = datasets.CIFAR10(data_dir, train=False, download=True, 
                            transform=transform_test)

loader_train = DataLoader(cifar10_train, batch_size=batch_size, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(cifar10_val, batch_size=batch_size, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
loader_test = DataLoader(cifar10_test, batch_size=batch_size)

it = iter(loader_test)
sample_inputs, _ = next(it)
fixed_input = sample_inputs[0:32, :, :, :]
save_image(denormalization(fixed_input), './CW/CAE/input_sample.png')

num_epochs = 30
learning_rate  = 0.001

#defining the model
hidden_size = 256 
def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True, negative_slope=0.2)]

def make_transconv_bn_relu(in_channels, out_channels, kernel_size=3, stride=2, padding=0):
    return [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)]

def make_transconv_tanh_final(in_channels, out_channels, kernel_size=3, stride=2, padding=0):
    return [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,  stride=stride, padding=padding, bias=False),
        nn.Tanh()]

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        

         
        #Downsampling part
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(3, 64, kernel_size=3, stride=2, padding=1 ))
        
        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=2, padding=1 ))
        
        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 16, kernel_size=3, stride=2, padding=1 ))
        
        self.up1 = nn.Sequential(
            *make_transconv_bn_relu(16, 32, kernel_size=2, stride=2))
        
        self.up2 = nn.Sequential(
            *make_transconv_bn_relu(32, 64, kernel_size=2, stride=2))
        
        self.up3 = nn.Sequential(
            *make_transconv_tanh_final(64, 3, kernel_size=2, stride=2))
        


    def encode(self, x):
        

        x = self.down1(x)
        #x = self.maxpool(x)
        #print(x.size())
        x = self.down2(x)
        #x = self.maxpool(x)
        #print(x.size())
        x = self.down3(x)
        #x = self.maxpool(x)
        #print(x.size())
        
        ### Here the image is encoded/compressed
        return x

    
    def decode(self, z):

        z=self.up1(z)
        #print(z.size())
        z=self.up2(z)
        #print(z.size())
        z=self.up3(z)
        #print(z.size())
        return z


    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

#define loss function
criterion = nn.MSELoss()  # can we use any other loss here? You are free to choose.
#criterion = nn.BCELoss()
def loss_function_CAE(recon_x, x):
    recon_loss = criterion(recon_x, x)
    return recon_loss

#Initialize Model and print number of parameters
model = CAE().to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
print(model)

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#train
train_losses = []
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, data in enumerate(loader_train):
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        # forward
        recon_batch = model(img)
        loss = loss_function_CAE(recon_batch, img)
        # backward
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    # print out losses and save reconstructions for every epoch
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, train_loss / len(loader_train)))
    recon = model(fixed_input.to(device))
    recon = denormalization(recon.cpu())
    save_image(recon, './CW/CAE/reconstructed_epoch_{}.png'.format(epoch))
    train_losses.append(train_loss/ len(loader_train))

# save the model and the loss values
np.save('./CW/CAE/train_losses.npy', np.array(train_losses))
torch.save(model.state_dict(), './CW/CAE/CAE_model2.pth')

#train loss curve

train_losses = np.load('./CW/CAE/train_losses.npy')
plt.plot(list(range(0,train_losses.shape[0])), train_losses)
plt.title('Train Loss')
plt.show()

#reconstruction
# load the model
model.load_state_dict(torch.load('./CW/CAE/CAE_model.pth'))
model.eval()
test_loss = 0
with torch.no_grad():
    for i, data in enumerate(loader_test):
        img,_ = data
        img = img.to(device)
        recon_batch = model(img)
        test_loss += loss_function_CAE(recon_batch, img)
    # loss calculated over the whole test set
    test_loss /= len(loader_test.dataset)
    print('Test set loss: {:.5f}'.format(test_loss))


model.load_state_dict(torch.load('./CW/CAE/CAE_model.pth'))
it = iter(loader_test)
sample_inputs, _ = next(it)
fixed_input = sample_inputs[0:32, :, :, :]

# visualize the original images of the last batch of the test set
img = make_grid(denormalization(fixed_input), nrow=8, padding=2, normalize=False,
                range=None, scale_each=False, pad_value=0)

show(img)

with torch.no_grad():
    # visualize the reconstructed images of the last batch of test set
    recon_batch = model(fixed_input.to(device)).cpu()
    recon_batch = make_grid(denormalization(recon_batch), nrow=8, padding=2, normalize=False,
                            range=None, scale_each=False, pad_value=0)
    show(recon_batch)

