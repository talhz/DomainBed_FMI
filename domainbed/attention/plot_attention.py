import torch
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import torchvision.transforms as transforms
from torchinfo import summary

from domainbed.algorithms import IRM
from domainbed import datasets
from domainbed import hparams_registry

import PIL.Image as Image
import numpy as np
from scipy.ndimage import gaussian_filter


torch.random.manual_seed(0)
np.random.seed(0)

hparams = hparams_registry.default_hparams('IRM', 'ColoredMNIST')
dataset = vars(datasets)['ColoredMNIST']('../data/MNIST', 2, hparams)

model_dict = torch.load('model_IRM_best.pkl', map_location=torch.device('cpu'))

model = IRM(model_dict['model_input_shape'], model_dict['model_num_classes'], model_dict['model_num_domains'], 
        model_dict['model_hparams'])

model.load_state_dict(model_dict['model_dict'])
featurizer = model.featurizer
network = model.network

class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.gradient = None 
        self.forward_output = None 
        
        self.hook_layers()
        
    def hook_layers(self):
        def forward_hook(module, input, output):
            self.forward_output = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]
            
        layer = dict([*self.model.named_modules()])[self.layer_name]
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)
        
    def generate_cam(self, input_image, target_class):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_image)
        pred = torch.argmax(output) # TODO: why is it always 1?
        target = output[:, pred]
        target.backward()
        
        gradient_pooled = torch.mean(self.gradient, [0, 2, 3])
        cam = torch.zeros(self.forward_output.shape[2:], dtype=torch.float32)
        
        for i in range(gradient_pooled.shape[0]):
            cam += gradient_pooled[i] * self.forward_output[0, i, :, :]
        
        cam = torch.clamp(cam, min=0)
        cam /= torch.max(cam)
        return cam, pred

def attention_plot(network, dataset, layer_name, row, col):
    def super_impose(network, layer_name, img, label):
        grad_cam = GradCAM(network, layer_name)
        cam, pred = grad_cam.generate_cam(img, label)
        cam = transforms.ToPILImage()(cam.unsqueeze(0))
        transform_resize = transforms.Resize(dataset.input_shape[1:], interpolation=transforms.InterpolationMode.BILINEAR)
        cam = transform_resize(cam)
        cam = transforms.ToTensor()(cam).squeeze()
        
        heatmap = cm.gray(cam)[:, :, :3]
        heatmap = np.uint8(heatmap * 255)
        img = np.uint8(img.squeeze(0).permute(1, 2, 0).numpy() * 255)
        superimposed_img = heatmap * 0.6 + img * 0.4 
        superimposed_img = np.uint8(superimposed_img)
        
        return superimposed_img, pred
    
    idx = np.random.choice(range(len(dataset[2])), row * col, replace=False)
    fig, axs = plt.subplots(row, col, figsize=(12, 8))
    for i in range(row * col):
        img, label = dataset[2][idx[i]]
        superimposed_img, pred = super_impose(network, layer_name, img.unsqueeze(0), label)
        axs[i // col, i % col].imshow(superimposed_img)
        axs[i // col, i % col].set_title(f'True label: {label}, Pred: {pred}')
        axs[i // col, i % col].axis('off')
    fig.suptitle('Attention Visualization: Testing Environment', fontsize=16)
    plt.show()
    
attention_plot(network, dataset, '0.conv4', 4, 4)
