import os
import sys
import argparse
import logging
import json
import time
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F

import torchvision
import torchvision.datasets as datasets
from data.utils import transform

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa

parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('--model_path', default='./', metavar='MODEL_PATH',
                    type=str, help="Path to the trained models")
parser.add_argument('--in_csv_path', default='dev.csv', metavar='IN_CSV_PATH',
                    type=str, help="Path to the input image path in csv")
parser.add_argument('--out_csv_path', default='test/test.csv',
                    metavar='OUT_CSV_PATH', type=str,
                    help="Path to the ouput predictions in csv")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")

if not os.path.exists('test'):
    os.mkdir('test')


def get_pred(output, cfg):
    if cfg.criterion == 'BCE' or cfg.criterion == "FL":
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
    elif cfg.criterion == 'CE':
        for num_class in cfg.num_classes:
            assert num_class >= 2
        prob = F.softmax(output)
        pred = prob[:, 1].cpu().detach().numpy()
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return pred


def test_epoch(cfg, args, ensemble, dataloader, out_csv_path,mnist_testset):
    torch.set_grad_enabled(False)
    for model in ensemble:
        model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    test_header = [
        'Path',
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Atelectasis',
        'Pleural Effusion']
    mode = 'OOD'
    with open(out_csv_path, 'w') as f:
        f.write(','.join(test_header) + '\n')
        
            
        for step in range(steps):
            
#             image = np.array(mnist_testset[step])
#             image = mnist_testset.data.numpy()[step]
#             image = transform(image, cfg)
#             image = torch.from_numpy(image)
#             image =  torch.reshape
#             image = np.random.normal(0,1,(8,3,400,400))
#             image = torch.from_numpy(image)
            
            
            image, path = next(dataiter)
            
            batch_size = len(path)
            images = np.zeros((batch_size,3,400,400))
            for i in range(batch_size):
                
                image = mnist_testset.data.numpy()[step*batch_size + i]
                image = transform(image, cfg,mode)
#                 image = torch.from_numpy(image)
                images[i] = image
            if mode == 'OOD':
                image = images 
                image = torch.from_numpy(image).float()
            np.random.seed(step)
            image = np.random.normal(0,1,(batch_size,3,400,400))
            image = torch.from_numpy(image).float()  
            image = image.to(device)
          
            outputs = []
            for model in ensemble:
                output, __ = model(image)
                outputs.append(output)
                
            
            pred = np.zeros((30, batch_size))
            
            for n, output in enumerate(outputs):
                for i in range(5):
#                     pred[i] += get_pred(output[i], cfg)
#                     a = torch.exp(output[:,i]).view(-1)
#                     b = torch.exp(output[:,i+5]).view(-1)
#                     pred[i] = a.cpu().detach().numpy()
#                     pred[i+5] = b.cpu().detach().numpy()
                
                    idx = n*5+i
                   
                    pred[idx] = torch.sigmoid(output[i]).squeeze().cpu().detach().numpy()
#                     pred[i] = torch.div(a,(a+b)).cpu().detach().numpy()
          
#             pred /= 6
            
            for i in range(batch_size):
                batch = ','.join(map(lambda x: '{}'.format(x), pred[:, i]))
                result = path[i] + ',' + batch
                f.write(result + '\n')
                logging.info('{}, Image : {}, Prob : {}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), path[i], batch))


def run(args):

        
    args.model_path += 'Ensemble-from_scratch/'
    with open(args.model_path + 'cfg.json') as f:
        cfg = edict(json.load(f))

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
            .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg,'a')
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    ensemble = []
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    
    for i in range(6):
        model = Classifier(cfg,mode='distill')
        model = DataParallel(model, device_ids=device_ids).to(device).eval()
#         model_name = 'best5.ckpt'
        model_name = 'best' + str(i+1) + '.ckpt'


        ckpt_path = os.path.join(args.model_path, model_name)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['state_dict'])
        ensemble.append(model)
 
        
    dataloader_test = DataLoader(
            ImageDataset(args.in_csv_path, cfg, mode='test'),
            batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
            drop_last=False, shuffle=False)

    test_epoch(cfg, args, ensemble, dataloader_test, args.out_csv_path,mnist_testset)

#     print('Save best is step :', ckpt['step'], 'AUC :', ckpt['auc_dev_best'])


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
