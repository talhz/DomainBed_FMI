import argparse

import torch
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import torchvision.transforms as transforms
from torchinfo import summary

from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry

import PIL.Image as Image
import numpy as np
from scipy.ndimage import gaussian_filter


LAYER_REGISTRY = {'ResNet': '0.network.layer4.2.conv2', 'MNIST_CNN': '0.conv4'}

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
        # print(dict([*self.model.named_modules()]))
        layer = dict([*self.model.named_modules()])[self.layer_name]
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)
        
    def generate_cam(self, input_image):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_image)
        pred = torch.argmax(output)
        target = output[:, pred]
        target.backward()
        
        gradient_pooled = torch.mean(self.gradient, [0, 2, 3])
        cam = torch.zeros(self.forward_output.shape[2:], dtype=torch.float32)
        
        for i in range(gradient_pooled.shape[0]):
            cam += gradient_pooled[i] * self.forward_output[0, i, :, :]
            
        cam = torch.clamp(cam, min=0)
        cam /= torch.max(cam)
        return cam, pred

def attention_plot(network, dataset, layer_name, row, col, label_dict=None):
    def super_impose(network, layer_name, img):
        grad_cam = GradCAM(network, layer_name)
        cam, pred = grad_cam.generate_cam(img)
        cam = transforms.ToPILImage()(cam.unsqueeze(0))
        transform_resize = transforms.Resize(dataset.input_shape[1:], interpolation=transforms.InterpolationMode.BILINEAR)
        cam = transform_resize(cam)
        cam = transforms.ToTensor()(cam).squeeze()
        heatmap = cm.gray(cam)[:, :, :3]
        heatmap = np.uint8(heatmap * 255)
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255
        superimposed_img = heatmap * 0.7 + 0.3 * img
        
        # print(superimposed_img)
        superimposed_img = np.uint8(superimposed_img)
        
        return superimposed_img, pred
    
    idx = np.random.choice(range(len(dataset[3])), row * col, replace=False)
    fig, axs = plt.subplots(row, col, figsize=(12, 8))
    for i in range(row * col):
        img, label = dataset[3][idx[i]]
        superimposed_img, pred = super_impose(network, layer_name, img.unsqueeze(0))
        axs[i // col, i % col].imshow(superimposed_img)
        if label_dict:
            label = label_dict[label]
            pred = label_dict[pred.item()]
        axs[i // col, i % col].set_title(f'True label: {label}, Pred: {pred}', fontsize=8)
        axs[i // col, i % col].axis('off')
    fig.suptitle('Attention Visualization: Training Environment', fontsize=16)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot attention map')
    parser.add_argument('--dataset', type=str, default='ColoredMNIST', help='dataset name')
    parser.add_argument('--data_dir', type=str, default='../data/MNIST', help='data directory')
    parser.add_argument('--algorithm', type=str, default='IRM', help='algorithm name')
    parser.add_argument('--seed', type=int, default=0, help='random seed for plotting image')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0], help='test environments')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--row', type=int, default=4, help='number of rows for plotting')
    parser.add_argument('--col', type=int, default=4, help='number of columns for plotting')
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
        
    # Only support default hyperparameter for now
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    
    if isinstance(dataset, datasets.MultipleEnvironmentImageFolder):
        label_dict = dataset.get_label_dict()
    else:
        label_dict = None
        
    model_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    model = algorithm_class(model_dict['model_input_shape'], model_dict['model_num_classes'], model_dict['model_num_domains'], 
            model_dict['model_hparams'])

    model.load_state_dict(model_dict['model_dict'])
    featurizer = model.featurizer
    network = model.network 
    layer_name = LAYER_REGISTRY[featurizer.name]
    attention_plot(network, dataset, layer_name, args.row, args.col, label_dict)
