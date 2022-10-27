from sklearn.ensemble import RandomForestClassifier
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


def denorm(x, channels=None, w=None, h=None, resize=False):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x


def show(img):
    if torch.cuda.is_available():
        img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx)
                          if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

# We set a random seed to ensure that your results are reproducible.
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
batch_size = 128

if not os.path.exists('./CW/CAE'):
    os.makedirs('./CW/CAE')
if not os.path.exists('./CW/DCGAN'):
    os.makedirs('./CW/DCGAN')


NUM_TRAIN = 49000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
save_image(denorm(fixed_input), './CW/CAE/input_sample.png')
num_epochs = 5
learning_rate = 0.001

hidden_size = 256


def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True, negative_slope=0.2)]


def make_transconv_bn_relu(in_channels, out_channels, kernel_size=3, stride=2, padding=0):
    return [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)]


def make_transconv_tanh_final(in_channels, out_channels, kernel_size=3, stride=2, padding=0):
    return [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, bias=False),
        nn.Tanh()]


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # Downsampling part
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(3, 64, kernel_size=3, stride=2, padding=1))

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=2, padding=1))

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 16, kernel_size=3, stride=2, padding=1))

        self.up1 = nn.Sequential(
            *make_transconv_bn_relu(16, 32, kernel_size=2, stride=2))

        self.up2 = nn.Sequential(
            *make_transconv_bn_relu(32, 64, kernel_size=2, stride=2))

        self.up3 = nn.Sequential(
            *make_transconv_tanh_final(64, 3, kernel_size=2, stride=2))

    def encode(self, x):

        x = self.down1(x)
        #x = self.maxpool(x)
        # print(x.size())
        x = self.down2(x)
        #x = self.maxpool(x)
        # print(x.size())
        x = self.down3(x)
        #x = self.maxpool(x)
        # print(x.size())

        # Here the image is encoded/compressed
        return x

    def decode(self, z):

        z = self.up1(z)
        # print(z.size())
        z = self.up2(z)
        # print(z.size())
        z = self.up3(z)
        # print(z.size())
        return z

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon


criterion = nn.MSELoss()
#criterion = nn.BCELoss()


def loss_function_CAE(recon_x, x):
    recon_loss = criterion(recon_x, x)
    return recon_loss


model = CAE().to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, train_loss / len(loader_train)))
    recon = model(fixed_input.to(device))
    recon = denorm(recon.cpu())
    save_image(recon, './CW/CAE/reconstructed_epoch_{}.png'.format(epoch))
    train_losses.append(train_loss / len(loader_train))


np.save('./CW/CAE/train_losses.npy', np.array(train_losses))
torch.save(model.state_dict(), './CW/CAE/CAE_model22.pth')

train_losses = np.load('./CW/CAE/train_losses.npy')
plt.plot(list(range(0, train_losses.shape[0])), train_losses)
plt.title('Train Loss')
plt.show()
model.load_state_dict(torch.load('./CW/CAE/CAE_model22.pth'))
model.eval()
test_loss = 0
with torch.no_grad():
    for i, data in enumerate(loader_test):
        img, _ = data
        img = img.to(device)
        recon_batch = model(img)
        test_loss += loss_function_CAE(recon_batch, img)
    # loss calculated over the whole test set
    test_loss /= len(loader_test.dataset)
    print('Test set loss: {:.5f}'.format(test_loss))

model.load_state_dict(torch.load('./CW/CAE/CAE_model2.pth'))
it = iter(loader_test)
sample_inputs, _ = next(it)
fixed_input = sample_inputs[0:32, :, :, :]


img = make_grid(denorm(fixed_input), nrow=8, padding=2, normalize=False,
                range=None, scale_each=False, pad_value=0)

show(img)
with torch.no_grad():

    recon_batch = model(fixed_input.to(device)).cpu()
    recon_batch = make_grid(denorm(recon_batch), nrow=8, padding=2, normalize=False,
                            range=None, scale_each=False, pad_value=0)
    show(recon_batch)
X_train, y_train = np.array([[]]), np.array([])
for i, data in enumerate(loader_train):
    img, classes = data
    img = img.to(device)
    img = model.encode(img).view(img.size(0), -1)
    if i == 0:
        X_train = img.cpu().detach().numpy()
    else:
        X_train = np.concatenate((X_train, img.cpu().detach().numpy()), axis=0)
    y_train = np.concatenate((y_train, classes), axis=0)

print(X_train.shape)
print(y_train.shape)


X_test, y_test = np.array([[]]), np.array([])
for i, data in enumerate(loader_test):
    img, classes = data
    img = img.to(device)
    img = model.encode(img).view(img.size(0), -1)
    if i == 0:
        X_test = img.cpu().detach().numpy()
    else:
        X_test = np.concatenate((X_test, img.cpu().detach().numpy()), axis=0)
    y_test = np.concatenate((y_test, classes), axis=0)

print(X_test.shape)
print(y_test.shape)
batch = []

random_features = torch.Tensor(65536).data.normal_(0, 0.5)
recon_batch = model.decode(random_features.view(
    256, 16, 4, 4).to(device)).cpu().detach().numpy()

recon_batch = recon_batch[0:32, :, :, :]
recon_batch = make_grid(denorm(torch.Tensor(recon_batch)), nrow=8, padding=2, normalize=False,
                        range=None, scale_each=False, pad_value=0)
show(recon_batch)
clf = RandomForestClassifier(
    n_estimators=100, max_depth=15, n_jobs=-1, random_state=0)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
num_epochs = 100
learning_rate = 0.0002
latent_vector_size = 100


def make_conv_sig_final(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.Sigmoid()]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.up1 = nn.Sequential(
            *make_transconv_bn_relu(latent_vector_size, 256, kernel_size=4, stride=1, padding=0))

        self.up2 = nn.Sequential(
            *make_transconv_bn_relu(256, 128, kernel_size=4, stride=2, padding=1))

        self.up3 = nn.Sequential(
            *make_transconv_bn_relu(128, 64, kernel_size=4, stride=2, padding=1))

        self.up4 = nn.Sequential(
            *make_transconv_tanh_final(64, 3, kernel_size=4, stride=2, padding=1))

    def decode(self, z):

        # print(z.size())
        z = self.up1(z)
        # print(z.size())
        z = self.up2(z)
        # print(z.size())
        z = self.up3(z)
        # print(z.size())
        z = self.up4(z)
        # print(z.size())

        return z

    def forward(self, z):
        return self.decode(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(3, 64, kernel_size=4, stride=2, padding=1))

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64, 128, kernel_size=4, stride=2, padding=1))

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=4, stride=2, padding=1))

        self.down4 = nn.Sequential(
            *make_conv_sig_final(256, 1, kernel_size=4, stride=1, padding=0))

    def discriminator(self, x):

        # print(x.size())
        x = self.down1(x)
        # print(x.size())
        x = self.down2(x)
        # print(x.size())
        x = self.down3(x)
        # print(x.size())
        out = self.down4(x)
        # print(out.size())

        return out

    def forward(self, x):
        out = self.discriminator(x)
        return out.view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


use_weights_init = True

model_G = Generator().to(device)
if use_weights_init:
    model_G.apply(weights_init)
params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
print("Total number of parameters in Generator is: {}".format(params_G))
print(model_G)
print('\n')

model_D = Discriminator().to(device)
if use_weights_init:
    model_D.apply(weights_init)
params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
print("Total number of parameters in Discriminator is: {}".format(params_D))
print(model_D)
print('\n')

print("Total number of parameters is: {}".format(params_G + params_D))

criterion = nn.BCELoss()


def loss_function(out, label):
    loss = criterion(out, label)
    return loss


beta1 = 0.5
optimizerD = torch.optim.Adam(
    model_D.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(
    model_G.parameters(), lr=learning_rate, betas=(beta1, 0.999))

fixed_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
real_label = 1
fake_label = 0

export_folder = './CW/DCGAN'
train_losses_G = []
train_losses_D = []

for epoch in range(num_epochs):
    for i, data in enumerate(loader_train, 0):
        train_loss_D = 0
        train_loss_G = 0
        
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        
        # train with real
        model_D.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = model_D(real_cpu)
        errD_real = loss_function(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, latent_vector_size,
                            1, 1, device=device)
        fake = model_G(noise)
        label.fill_(fake_label)
        output = model_D(fake.detach())
        errD_fake = loss_function(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        train_loss_D += errD.item()
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))

        model_G.zero_grad()
        label.fill_(real_label)
        output = model_D(fake)
        errG = loss_function(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        train_loss_G += errG.item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(loader_train),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    if epoch == 0:
        save_image(denorm(real_cpu.cpu()), './CW/DCGAN/real_samples.png')

    fake = model_G(fixed_noise)
    save_image(denorm(fake.cpu()),
               './CW/DCGAN/fake_samples_epoch_%03d.png' % epoch)
    train_losses_D.append(train_loss_D / len(loader_train))
    train_losses_G.append(train_loss_G / len(loader_train))

# save losses and models
np.save('./CW/DCGAN/train_losses_D.npy', np.array(train_losses_D))
np.save('./CW/DCGAN/train_losses_G.npy', np.array(train_losses_G))
torch.save(model_G.state_dict(), './CW/DCGAN/DCGAN_model_G2.pth')
torch.save(model_D.state_dict(), './CW/DCGAN/DCGAN_model_D2.pth')

it = iter(loader_test)
sample_inputs, _ = next(it)
fixed_input = sample_inputs[0:32, :, :, :]

# visualize the original images of the last batch of the test set
img = make_grid(denorm(fixed_input), nrow=8, padding=2, normalize=False,
                range=None, scale_each=False, pad_value=0)
show(img)
model_G.load_state_dict(torch.load('./CW/DCGAN/DCGAN_model_G2.pth'))
input_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)

with torch.no_grad():
    # visualize the generated images
    generated = model_G(input_noise).cpu()
    generated = make_grid(denorm(generated)[:32], nrow=8, padding=2, normalize=False,
                          range=None, scale_each=False, pad_value=0)
    show(generated)

train_losses_D = np.load('./CW/DCGAN/train_losses_D.npy')
train_losses_G = np.load('./CW/DCGAN/train_losses_G.npy')
plt.plot(
    list(range(0, train_losses_D.shape[0])), train_losses_D, label='loss_D')
plt.plot(
    list(range(0, train_losses_G.shape[0])), train_losses_G, label='loss_G')
plt.legend()
plt.title('Train Losses')
plt.show()
