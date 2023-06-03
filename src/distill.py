import sys
import os
import argparse
import logging
import json
import time
import subprocess
from shutil import copyfile

import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.dataset_distill import ImageDataset  # noqa
from model.classifier import Classifier  # noqa
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa
from scipy.special import gamma
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', default='distillation', metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train', default=None, type=str, help="If get"
                    "parameters from pretrained model")
parser.add_argument('--resume', default=0, type=int, help="If resume from "
                    "previous run")
parser.add_argument('--logtofile', default=False, type=bool, help="Save log "
                    "in save_path/log.txt if set True")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")

class MLE_Loss(torch.nn.Module):
    
    def __init__(self):
        super(MLE_Loss,self).__init__()
        
    def forward(self,output,target_u, device, cfg, mode , T,gamma):

        target_s = torch.add(torch.mul((1-gamma),target_u), 0.5*gamma)
        logit_t = torch.log(torch.div(target_s,(1-target_s)))
        target = torch.sigmoid(torch.div(logit_t,T))


        target_shape = torch.FloatTensor([8,5,6])
        alphas_shape = torch.FloatTensor([8,5])
        reshape_params = [8,5,6]

        alphas = torch.exp(torch.div(output[:,0:5],T))
        betas = torch.exp(torch.div(output[:,5:10],T))

        alpha_0s = torch.add(alphas, betas)
        ##log(gamma(alpha+beta))

        lgamma1 = alpha_0s.lgamma()
        term1s = torch.div(lgamma1,alphas_shape[0])
        term1ss = torch.div(term1s,alphas_shape[1])
        term1 = torch.sum(term1ss)
        ##log(gamma(alpha)) 

        lgamma2 = alphas.lgamma()
        term2s = torch.div(lgamma2,alphas_shape[0])
        term2ss = torch.div(term2s,alphas_shape[1])
        term2 = torch.sum(term2ss)

        ##log(gamma(beta))

        lgamma3 = betas.lgamma()
        term3s = torch.div(lgamma3,alphas_shape[0])
        term3ss = torch.div(term3s,alphas_shape[1])
        term3 = torch.sum(term3ss)

        ## alpha-1
        a = torch.reshape((alphas - 1),(1,reshape_params[0]*reshape_params[1]))
        ## y
        b = torch.reshape(torch.log(target),(reshape_params[0]*reshape_params[1],reshape_params[2]))
        ## (alpha-1) x y

        term4 = torch.sum(torch.div(torch.matmul(a,b),240))


        ## (beta-1)
        c = torch.reshape(torch.add(betas, -1),(1,reshape_params[0]*reshape_params[1]))
        ## (1-y)
        d = torch.reshape(torch.log(torch.add(1,-target)),(reshape_params[0]*reshape_params[1],reshape_params[2]))
        ## (beta-1) x (1-y)

        term5 = torch.sum(torch.div(torch.matmul(c,d),240))
        ## -log(gamma(alpha+beta))+log(gamma(alpha))+log(gamma(beta))-((alpha-1)y)-((beta-1)(1-y)) 
        l12 = torch.add(-term1,term2)
        l34 = torch.add(term3,-term4)
        l1234 = torch.add(l12,l34)
        loss = torch.add(l1234,-term5)
        
        
        if mode == 'test':
        ## print predictions and ensemble mean from last batch in validation set
            print('last batch loss : ',loss)
            for i in range(8):
                preds = alphas[i]/(alphas[i]+betas[i])
                target_means = torch.mean(target[i], axis=1)


                print('predictions : ', preds.squeeze())
                print('target_means : ', target_means)
                if i == 7:
                    a = alphas[i].squeeze().squeeze()
                    b = betas[i].squeeze().squeeze()
                    print('alphas : ', a)
                    print('betas : ', b)
                    
        return loss





def train_epoch(summary, summary_dev, cfg, args, model, dataloader,
                dataloader_dev, optimizer, summary_writer, best_dict,
                dev_header,finish_step,epoch,Loss_Func):
    torch.set_grad_enabled(True)
    model.train()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    label_header = dataloader.dataset._label_header
    num_tasks = len(cfg.num_classes)

    time_now = time.time()
#     loss_sum = np.zeros(num_tasks)
#     acc_sum = np.zeros(num_tasks)
    loss_sum = 0 
    T = 1
    if epoch == 0:
        T=5
    if epoch == 1:
        T=3
    
    T = torch.tensor(float(T))
    T = T.to(device)
    gamma = 1e-4
    
    gamma = torch.tensor(float(gamma))
    gamma = gamma.to(device)
    

    
    for step in range(steps):
        image, target = next(dataiter)
        
        image = image.to(device)
        target = target.to(device)
#         target = Variable(target.squeeze(), requires_grad=True)
        target = target.squeeze()
        
        output, logit_map = model(image)
        output.retain_grad()

        loss = Loss_Func.forward(output,target, device, cfg, 'train', T, gamma)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        optimizer.step()
        loss_sum += loss.item()

        summary['step'] += 1
        
        if (epoch == 0) and (step<100):
            summary['lcurve_t100'].append(str(loss))

        if summary['step'] % cfg.log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg.log_every
#             acc_sum /= cfg.log_every
#             loss_str = ' '.join(lambda x: '{:.5f}'.format(x), loss_sum)
            loss_str = str(loss_sum)
#             acc_str = ' '.join(map(lambda x: '{:.3f}'.format(x), acc_sum))
#             summary['lcurve_t'].append(loss_str)

            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Loss : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], loss_str, time_spent))

#             for t in range(num_tasks):
#                 summary_writer.add_scalar(
#                     'train/loss_{}'.format(label_header[t]), loss_sum[t],
#                     summary['step'])
#                 summary_writer.add_scalar(
#                     'train/acc_{}'.format(label_header[t]), acc_sum[t],
#                     summary['step'])

#             loss_sum = np.zeros(num_tasks)
#             acc_sum = np.zeros(num_tasks)
            loss_sum = 0
            

        if summary['step'] % cfg.test_every == 0:

            print('Temperature : ',T)
            time_now = time.time()
            summary_dev = test_epoch(
                summary_dev, cfg, args, model, dataloader_dev, Loss_Func)
            time_spent = time.time() - time_now

#             auclist = []
#             for i in range(len(cfg.num_classes)):
#                 y_pred = predlist[i]
#                 y_true = true_list[i]
#                 fpr, tpr, thresholds = metrics.roc_curve(
#                     y_true, y_pred, pos_label=1)
#                 auc = metrics.auc(fpr, tpr)
#                 auclist.append(auc)
#             summary_dev['auc'] = np.array(auclist)

#             loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
#                                         summary_dev['loss']))
            loss_dev_str = str(summary_dev['loss'])
            if step % 2000:
                summary['lcurve_v'].append(loss_dev_str)
#             acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
#                                        summary_dev['acc']))
#             auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
#                                        summary_dev['auc']))

            logging.info(
                '{}, Dev, Step : {}, Loss : {},'
                'Run Time : {:.2f} sec' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary['step'],
                    loss_dev_str,
                    time_spent))

#             for t in range(len(cfg.num_classes)):
            summary_writer.add_scalar(
                'dev/loss_',
                summary_dev['loss'], summary['step'])
#                 summary_writer.add_scalar(
#                     'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
#                     summary['step'])
#                 summary_writer.add_scalar(
#                     'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
#                     summary['step'])

            save_best = False
#             mean_acc = summary_dev['acc'][cfg.save_index].mean()
#             if mean_acc >= best_dict['acc_dev_best']:
#                 best_dict['acc_dev_best'] = mean_acc
#                 if cfg.best_target == 'acc':
#                     save_best = True

#             mean_auc = summary_dev['auc'][cfg.save_index].mean()
#             if mean_auc >= best_dict['auc_dev_best']:
#                 best_dict['auc_dev_best'] = mean_auc
#                 if cfg.best_target == 'auc':
#                     save_best = True

            mean_loss = summary_dev['loss']
            if (mean_loss <= best_dict['loss_dev_best']) and (mean_loss > -10):
                best_dict['loss_dev_best'] = mean_loss
#                 if cfg.best_target == 'loss':
                save_best = True
                    

            if save_best:
                torch.save(
                    {'epoch': summary['epoch'],
                     'step': summary['step'],
#                      'acc_dev_best': best_dict['acc_dev_best'],
#                      'auc_dev_best': best_dict['auc_dev_best'],
                     'loss_dev_best': best_dict['loss_dev_best'],
                     'lcurve_t': summary['lcurve_t'],
                     'lcurve_v': summary['lcurve_v'],
                     'lcurve_t100': summary['lcurve_t100'],
                     'state_dict': model.module.state_dict()},
                    os.path.join(args.save_path, 'best{}.ckpt'.format(
                        best_dict['best_idx']))
                )
                best_dict['best_idx'] += 1
                if best_dict['best_idx'] > cfg.save_top_k:
                    best_dict['best_idx'] = 1
                logging.info(
                    '{}, Best, Step : {}, Loss : {},'
                    'Best Loss : {:.3f}' .format(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['step'],
                        loss_dev_str,
#                         acc_dev_str,
#                         auc_dev_str,
                        best_dict['loss_dev_best']))
        model.train()
        torch.set_grad_enabled(True)
    summary['epoch'] += 1

    return summary, best_dict


def test_epoch(summary, cfg, args, model, dataloader, Loss_Func):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)
    T=1
#     loss_sum = np.zeros(num_tasks)
#     acc_sum = np.zeros(num_tasks)
    loss_sum = 0
    gamma = 1e-4

#     predlist = list(x for x in range(len(cfg.num_classes)))
#     true_list = list(x for x in range(len(cfg.num_classes)))
    
    for step in range(steps):
        if step == steps-1:
            mode = 'test'
        else:
            mode = None
        image, target = next(dataiter)
        target = target.squeeze()
        image = image.to(device)
        target = target.to(device)
        output, logit_map = model(image)
        # different number of tasks
#         loss = get_loss(output, target, device, cfg, mode)
        
        loss = Loss_Func.forward(output,target, device, cfg, mode, T, gamma)
        loss_sum += loss
#         for t in range(len(cfg.num_classes)):

#             loss_t, acc_t = get_loss(output, target, t, device, cfg)
#             # AUC
#             output_tensor = torch.sigmoid(
#                 output[t].view(-1)).cpu().detach().numpy()
#             target_tensor = target[:, t].view(-1).cpu().detach().numpy()
#             if step == 0:
#                 predlist[t] = output_tensor
#                 true_list[t] = target_tensor
#             else:
#                 predlist[t] = np.append(predlist[t], output_tensor)
#                 true_list[t] = np.append(true_list[t], target_tensor)

#             loss_sum[t] += loss_t.item()
#             acc_sum[t] += acc_t.item()
    summary['loss'] = loss_sum / steps
#     summary['acc'] = acc_sum / steps

    return summary


def run(args):
#     print(args.cfg_path)
#     print('SAVE PATH : ',args.save_path)
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
#         print(cfg)
        if args.verbose is True:
            print(json.dumps(cfg, indent=4))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if args.logtofile is True:
        logging.basicConfig(filename=args.save_path + '/log.txt',
                            filemode="w", level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
            .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg)
    
    if args.verbose is True:
        from torchsummary import summary
        if cfg.fix_ratio:
            h, w = cfg.long_side, cfg.long_side
        else:
            h, w = cfg.height, cfg.width
        summary(model.to(device), (3, h, w))
    model = DataParallel(model, device_ids=device_ids).to(device).train()
    
    print('PRE-TRAIN : ',args.pre_train)

    if args.pre_train is not None: 
        print('PATH EXISTS : ' ,os.path.exists(args.pre_train))
        if os.path.exists(args.pre_train):
            if args.pre_train[-5:] =='.ckpt':
                print('HERE2!')
                ckpt = torch.load(args.pre_train, map_location=device)
                model.module.load_state_dict(ckpt['state_dict'])
                     
            else:
                ckpt = torch.load(args.pre_train, map_location=device)
                model.module.load_state_dict(ckpt)
                print('HERE1!')
            
           
#         else:
#             model = Classifier(cfg)
#             model = DataParallel(model, device_ids=device_ids).to(device).train()
#             ckpt_path = os.path.join(args.model_path, 'resume.ckpt')
#             ckpt = torch.load(ckpt_path, map_location=device)
#             model.module.load_state_dict(ckpt['state_dict'])
#             print('HERE2!')
#             print('WEIGHTS : ',model.get_weights())
            
    optimizer = get_optimizer(model.parameters(), cfg)

    src_folder = os.path.dirname(os.path.abspath(__file__)) + '/../'
    dst_folder = os.path.join(args.save_path, 'classification')
    rc, size = subprocess.getstatusoutput('du --max-depth=0 %s | cut -f1'
                                          % src_folder)
    if rc != 0:
        raise Exception('Copy folder error : {}'.format(rc))
    rc, err_msg = subprocess.getstatusoutput('cp -R %s %s' % (src_folder,
                                                              dst_folder))
    if rc != 0:
        raise Exception('copy folder error : {}'.format(err_msg))

    copyfile(cfg.train_csv, os.path.join(args.save_path, 'train.csv'))
    copyfile(cfg.dev_csv, os.path.join(args.save_path, 'dev.csv'))

    dataloader_train = DataLoader(
        ImageDataset(cfg.train_csv, cfg, mode='train'),
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=True, shuffle=True)
    dataloader_dev = DataLoader(
        ImageDataset(cfg.dev_csv, cfg, mode='dev'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)
    dev_header = dataloader_dev.dataset._label_header

    summary_train = {'epoch': 0, 'step': 0, 'lcurve_t': [], 'lcurve_v': [], 'lcurve_t100': []}
    summary_dev = {'loss': float('inf'), 'acc': 0.0}
    print(args.save_path)
    summary_writer = SummaryWriter(args.save_path)
    epoch_start = 0
    best_dict = {
        "acc_dev_best": 0.0,
        "auc_dev_best": 0.0,
        "loss_dev_best": float('inf'),
        "fused_dev_best": 0.0,
        "best_idx": 1}
    
    finish_step = 0
    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'train.ckpt')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['state_dict'])
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step'], 'lcurve_t': ckpt['lcurve_t'], 'lcurve_v': ckpt['lcurve_v'],
                        'lcurve_t100': ckpt['lcurve_t100']}
#         best_dict['acc_dev_best'] = ckpt['acc_dev_best']
        best_dict['loss_dev_best'] = ckpt['loss_dev_best']
#         best_dict['auc_dev_best'] = ckpt['auc_dev_best']
        
        finish_step = 0
   
    
            
        
    Loss_Func = MLE_Loss()
    for epoch in range(epoch_start, cfg.epoch):
        print('epoch : ',epoch)
        lr = lr_schedule(cfg.lr, cfg.lr_factor, summary_train['epoch'],
                         cfg.lr_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
            

        summary_train, best_dict = train_epoch(
            summary_train, summary_dev, cfg, args, model,
            dataloader_train, dataloader_dev, optimizer,
            summary_writer, best_dict, dev_header, finish_step,epoch,Loss_Func)

        time_now = time.time()
        summary_dev = test_epoch(
            summary_dev, cfg, args, model, dataloader_dev,Loss_Func)
        time_spent = time.time() - time_now

#         auclist = []
#         for i in range(len(cfg.num_classes)):
#             y_pred = predlist[i]
#             y_true = true_list[i]
#             fpr, tpr, thresholds = metrics.roc_curve(
#                 y_true, y_pred, pos_label=1)
#             auc = metrics.auc(fpr, tpr)
#             auclist.append(auc)
#         summary_dev['auc'] = np.array(auclist)

#         loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
#                                     summary_dev['loss']))
    
        loss_dev_str = str(summary_dev['loss'])
#         acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
#                                    summary_dev['acc']))
#         auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
#                                    summary_dev['auc']))

        logging.info(
            '{}, Dev, Step : {}, Loss : {},  '
            'Run Time : {:.2f} sec' .format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                summary_train['step'],
                loss_dev_str,
                
                
                time_spent))

        
        summary_writer.add_scalar(
            'dev/loss_{}'.format(dev_header), summary_dev['loss'],
            summary_train['step'])
#             summary_writer.add_scalar(
#                 'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
#                 summary_train['step'])
#             summary_writer.add_scalar(
#                 'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
#                 summary_train['step'])

        save_best = False

#         mean_acc = summary_dev['acc'][cfg.save_index].mean()
#         if mean_acc >= best_dict['acc_dev_best']:
#             best_dict['acc_dev_best'] = mean_acc
#             if cfg.best_target == 'acc':
#                 save_best = True

#         mean_auc = summary_dev['auc'][cfg.save_index].mean()
#         if mean_auc >= best_dict['auc_dev_best']:
#             best_dict['auc_dev_best'] = mean_auc
#             if cfg.best_target == 'auc':
#                 save_best = True

        mean_loss = summary_dev['loss']
        if (mean_loss <= best_dict['loss_dev_best']) and (mean_loss > -2):
            best_dict['loss_dev_best'] = mean_loss
#             if cfg.best_target == 'loss':
            save_best = True

        if save_best:
            torch.save(
                {'epoch': summary_train['epoch'],
                 'step': summary_train['step'],
                 
                 
                 'loss_dev_best': best_dict['loss_dev_best'],
                 'lcurve_t': summary_train['lcurve_t'],
                 'lcurve_v': summary_train['lcurve_v'],
                 'lcurve_t100': summary_train['lcurve_t100'],
                 'state_dict': model.module.state_dict()},
                os.path.join(args.save_path,
                             'best{}.ckpt'.format(best_dict['best_idx']))
            )
            best_dict['best_idx'] += 1
            if best_dict['best_idx'] > cfg.save_top_k:
                best_dict['best_idx'] = 1
            logging.info(
                '{}, Best, Step : {}, Loss : {}, '
                 .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['step'],
                    loss_dev_str))
                
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'loss_dev_best': best_dict['loss_dev_best'],
                    'lcurve_t': summary_train['lcurve_t'],
                    'lcurve_v': summary_train['lcurve_v'],
                    'lcurve_t100': summary_train['lcurve_t100'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))
    summary_writer.close()


def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)
    print('DONE')


if __name__ == '__main__':
    main()
