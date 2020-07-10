import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import os

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import pygifsicle

os.makedirs('results', exist_ok = True)

image_size = 512
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(means, stds)])

def load_image(path, transform):
    image = Image.open(path)
    image = transform(image)
    return image

content_image = load_image('assets/dog.jpg', transform)
style_image = load_image('assets/starry-night.jpg', transform)

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def show_image(image):
    image = image.detach().clone()
    image = normalize_image(image)
    image = image.cpu().permute(1,2,0).numpy()
    plt.imshow(image)

def save_image(name, image):
    image = image.detach().clone()
    image = normalize_image(image)
    image = image.cpu().permute(1,2,0).numpy()
    plt.imsave(name, image)

def get_content_loss(features, target):
    loss = F.mse_loss(features, target.detach())
    return loss

def get_gram_matrix(x):
    b, c, h, w = x.shape
    x = x.view(b * c, h * w)
    g = torch.mm(x, x.t())
    g = g.div(b * c * h * w)
    return g

def get_style_loss(features, target):
    g_features = get_gram_matrix(features)
    g_target = get_gram_matrix(target)
    loss = F.mse_loss(g_features, g_target.detach())
    return loss

def get_tv_loss(x):
    diff_1 = x[:, :, :, 1:] - x[:, :, :, :-1]
    diff_2 = x[:, :, 1:, :] - x[:, :, :-1, :]
    var_1 = torch.sum(torch.abs(diff_1))
    var_2 = torch.sum(torch.abs(diff_2))
    return var_1 + var_2

vgg = models.vgg19(pretrained = True).features.eval()

class VGG(nn.Module):
    def __init__(self, vgg, content_layers, style_layer):
        super().__init__()
        assert isinstance(vgg, nn.Sequential)
        self.content_layers = set(content_layers)
        self.style_layers = set(style_layers)
        max_layers = max(max(content_layers), max(style_layers))
        self.vgg = vgg[:max_layers+1]

    def forward(self, x):
        content_features = []
        style_features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in content_layers:
                content_features.append(x)
            if i in style_layers:
                style_features.append(x)
        return content_features, style_features

content_layers = [7]
style_layers = [0, 2, 5, 7, 10]

model = VGG(vgg, content_layers, style_layers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
content_image = content_image.to(device)
style_image = style_image.to(device)

input_image = content_image.clone().requires_grad_()

# uncomment if want to generate image starting from random noise
#input_image = content_image.clone().normal_().requires_grad_()

optimizer = optim.LBFGS([input_image])

content_image = content_image.unsqueeze(0)
style_image = style_image.unsqueeze(0)
input_image = input_image.unsqueeze(0)

n_steps = 100
content_weight = 1
style_weight = 1e6
tv_weight = 1e-6

for i in tqdm(range(n_steps)):

    def closure():

        optimizer.zero_grad()

        content_content_features, content_style_features = model(content_image)
        style_content_features, style_style_features = model(style_image)
        input_content_features, input_style_features = model(input_image)

        content_loss = 0

        for input_features, content_features in zip(input_content_features, content_content_features):
            content_loss += get_content_loss(input_features, content_features)

        style_loss = 0

        for input_features, style_features in zip(input_style_features, style_style_features):
            style_loss += get_style_loss(input_features, style_features)

        tv_loss = get_tv_loss(input_image)

        content_loss = content_weight * content_loss
        style_loss = style_weight * style_loss
        tv_loss = tv_weight * tv_loss

        loss = content_loss + style_loss + tv_loss

        loss.backward()

        return loss

    optimizer.step(closure)

    show_image(input_image.squeeze())
    save_image(f'results/image-{i+1}.png', input_image.squeeze())

images = []

file_names = [f'results/image-{i+1}.png' for i in range(n_steps)]

for file_name in file_names:
    images.append(imageio.imread(file_name))
imageio.mimsave('results/image-animated.gif', images)

pygifsicle.optimize('results/image-animated.gif')
