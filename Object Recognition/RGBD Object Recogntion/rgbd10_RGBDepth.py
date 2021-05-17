from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo

import matplotlib
from matplotlib.offsetbox import AnchoredText
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time, math
import copy
import os,errno
# import bisect
from operator import itemgetter
from discriminative_dann import *
from depth_gen import *
import imagefolder
import argparse
from PIL import Image, ImageDraw,ImageFont

import torchvision.utils as vutils
######################################################################
# Load Data
torch.backends.cudnn.benchmark = True
CLS_EP = 0
POS_CLS_EP = 0
# POS_CLS_EP = 20
RGB_Ratio = 1.0
Depth_Ratio = 0.2
DEPTH_GT = True
USE_COFUSE = True
USE_Conditional = True
USE_Half = True
USE_INDI = True

parser = argparse.ArgumentParser(description='Domain Adaptation')
parser.add_argument('--source_set', type=str, default='amazon')
parser.add_argument('--target_set', type=str, default='webcam')

parser.add_argument('--gpu', type=str, default='2')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_class', type=int, default=31)
parser.add_argument('--base_lr', type=float, default=0.0015)

parser.add_argument('--pretrain_sample', type=int, default=50000)
parser.add_argument('--train_sample', type=int, default=200000)

parser.add_argument('--form_w', type=float, default=0.4)
parser.add_argument('--main_w', type=float, default=-0.8)

parser.add_argument('--form_w2', type=float, default=0.4)
parser.add_argument('--main_w2', type=float, default=-0.8)

parser.add_argument('--wp', type=float, default=0.055)
parser.add_argument('--wt', type=float, default=1)

parser.add_argument('--select', type=str, default='1-2')

parser.add_argument('--usePreT2D', type=bool, default=False)

parser.add_argument('--useT1DorT2', type=str, default="T2")

parser.add_argument('--diffS', type=bool, default=False)

parser.add_argument('--diffDFT2', type=bool, default=False)

parser.add_argument('--useT2CompD', type=bool, default=False)
parser.add_argument('--usemin', type=bool, default=False)

parser.add_argument('--useRatio', type=bool, default=False)

parser.add_argument('--useCurrentIter', type=bool, default=False)
# parser.add_argument('--useEpoch', type=bool, default=False)

parser.add_argument('--useLargeLREpoch', type=bool, default=True)

parser.add_argument('--MaxStep', type=int, default=0)

parser.add_argument('--useSepTrain', type=bool, default=True)

parser.add_argument('--fixW', type=bool, default=False)
parser.add_argument('--decay', type=float, default=0.0003)
parser.add_argument('--nesterov', type=bool, default=False)

parser.add_argument('--ReTestSource', type=bool, default=False)

parser.add_argument('--sourceTestIter', type=int, default=2000)
parser.add_argument('--defaultPseudoRatio', type=int, default=0.2)
parser.add_argument('--totalPseudoChange', type=int, default=100)
parser.add_argument('--usePrevAcc', type=bool, default=False)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--epochs2', type=int, default=100)

parser.add_argument('--usingTriDec', type=bool, default=False)
parser.add_argument('--usingDoubleDecrease', type=bool, default=True)
parser.add_argument('--usingTriDec2', type=bool, default=False)
parser.add_argument('--usingDoubleDecrease2', type=bool, default=True)

parser.add_argument('-g','--lrG', type=float, default=0.003, help='Glearning rate')
parser.add_argument('-d','--lrD', type=float, default=0.0003, help='Dlearning rate')
parser.add_argument('-b','--beta1', type=float, default=0.5, help='beta1 for adam')

args = parser.parse_args()

data_dir = '/home/wzha8158/datasets/RGBD_Square_Small/'
save_dir = './models/'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

######################################################################
# Load Data

# source_set = 'webcam'
# target_set = 'amazon'
#
# source_set = 'Bremen'
# target_set = 'Washington'

print('RGBD-10: ' + args.source_set + ' To ' + args.target_set)

data_transforms = {
    'source': transforms.Compose([
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'target': transforms.Compose([
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'depth': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'depth_test': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

# s_dir = '/home/wzha8158/datasets/RGBD/'

source_rgb_dir = data_dir+args.source_set+'/Square_RGB_256_Small/'
# _256
source_depth_dir = data_dir+args.source_set+'/Equalized_Square_Color_256_Small/'
# _256
# source_depth_dir = data_dir+source_set+'/DEPTH_Gray/'
 
target_rgb_dir = data_dir+args.target_set+'/Square_RGB_256_Small/'
# _256
target_depth_dir = data_dir+args.target_set+'/Equalized_Square_Color_256_Small/'
# _256

# if source_set == 'Caltech':
#     source_rgb_dir = data_dir+'Caltech/'
#     source_depth_dir = data_dir+'Caltech/'
#     target_rgb_dir = data_dir+'RGBD/RGB/'
#     target_depth_dir = data_dir+'RGBD/DEPTH/'

# source_rgb_dir = data_dir+'Washington/RGB/'
# source_depth_dir = data_dir+'Washington/Depth_Color/'
# target_rgb_dir = data_dir+'Caltech/RGB/'
# target_depth_dir = data_dir+'Caltech/RGB/'

dsets = {}

# ---------------------- read rgb images --------------------------
# dsets['source'] = imagefolder.ImageFolder(source_rgb_dir, source_depth_dir, 
#                                           data_transforms['source'], data_transforms['depth'])
# dsets['target'] = imagefolder.ImageFolder(target_rgb_dir, target_depth_dir, 
#                                           data_transforms['target'], data_transforms['depth'])
# dsets['test'] = imagefolder.ImageFolder(target_rgb_dir, target_depth_dir, 
#                                         data_transforms['test'], data_transforms['depth_test'])

# dsets['pseudo'] = []
# # dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=int(BATCH_SIZE / 2),
# #                 shuffle=False) for x in ['source_rgb', 'target_rgb', 'pseudo_source_rgb', 'test']}
# dset_loaders = {}
# dset_loaders['source'] = torch.utils.data.DataLoader(dsets['source'], batch_size=int(BATCH_SIZE / 2), shuffle=True)
# dset_loaders['target'] = torch.utils.data.DataLoader(dsets['target'], batch_size=int(BATCH_SIZE / 2), shuffle=True)
# dset_loaders['test'] = torch.utils.data.DataLoader(dsets['test'], batch_size=1, shuffle=False)

# source_total_set = []
# target_total_set = []

dsets = {}
dsets['source'] = imagefolder.ImageFolder(source_rgb_dir,
                                          data_transforms['source'] )

dsets['d_source'] = imagefolder.ImageFolder(source_rgb_dir, 
                                          data_transforms['source'])

dsets['test_source'] = imagefolder.ImageFolder(source_rgb_dir, 
                                          data_transforms['test'])
# dsets['target_source'] = imagefolder.ImageFolder(data_dir+args.source_set, data_transforms['source'])

dsets['target'] = imagefolder.ImageFolder(target_rgb_dir,
                                          data_transforms['target'] )

dsets['target_test'] = imagefolder.ImageFolder(target_rgb_dir,
                                          data_transforms['test'])

if not DEPTH_GT:
    dsets['target_depth_gen'] = imagefolder.ImageFolder(target_rgb_dir, 
                                              data_transforms['target'])
else:
    dsets['target_depth_gen'] = imagefolder.DepthImageFolder(target_depth_dir, 
                                              data_transforms['depth'])

dsets['d_pseudo_source'] = imagefolder.ImageFolder(source_rgb_dir,
                                          data_transforms['source'])


dsets['source_depth'] = imagefolder.DepthImageFolder(source_depth_dir, 
                                          data_transforms['depth'])

dsets['d_source_depth'] = imagefolder.DepthImageFolder(source_depth_dir, 
                                          data_transforms['depth'])

dsets['test_source_depth'] = imagefolder.DepthImageFolder(source_depth_dir, 
                                          data_transforms['depth_test'])
# dsets['target_source'] = imagefolder.ImageFolder(data_dir+args.source_set, data_transforms['source'])


dsets['target_test_depth'] = imagefolder.DepthImageFolder(target_depth_dir, 
                                          data_transforms['depth_test'])

dsets['d_pseudo_source_depth'] = imagefolder.DepthImageFolder(source_depth_dir, 
                                          data_transforms['depth'])

dsets['pseudo'] = []
dsets['d_pseudo_target'] = []

dsets['d_pseudo'] = []
dsets['d_pseudo_feat'] = []

dsets['d_pseudo_all'] = []
dsets['d_pseudo_all_feat'] = [] 

dsets['pseudo_depth'] = []
dsets['d_pseudo_target_depth'] = []

dsets['d_pseudo_depth'] = []
dsets['d_pseudo_feat_depth'] = []

dsets['d_pseudo_all_depth'] = []
dsets['d_pseudo_all_feat_depth'] = [] 

dsets['target_test_total'] = imagefolder.TotalImageFolder(target_rgb_dir, target_depth_dir, data_transforms['test'],
                                          data_transforms['depth_test'])

dsets['test'] = imagefolder.TotalImageFolder(target_rgb_dir, target_depth_dir, data_transforms['test'],
                                          data_transforms['depth_test'])

dsets['depth_gen'] = imagefolder.TotalImageFolder(source_rgb_dir, source_depth_dir , data_transforms['test'],
                                          data_transforms['depth_test'])

dsets['depth_gen_target'] = imagefolder.TotalImageFolder(target_rgb_dir, target_depth_dir , data_transforms['test'],
                                          data_transforms['depth_test'])

dsets['depth_gen_draw'] = imagefolder.TotalImageFolder(target_rgb_dir, target_depth_dir , data_transforms['test'],
                                          data_transforms['depth_test'])

dsets['draw'] = imagefolder.TotalImageFolder(target_rgb_dir, target_depth_dir, data_transforms['test'],
                                          data_transforms['depth_test'])


# dsets['test_depth'] = imagefolder.DepthImageFolder(target_depth_dir, data_transforms['depth_test'])

# dsets['draw_depth'] = imagefolder.DepthImageFolder(target_depth_dir, data_transforms['depth_test'])


dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=int(args.batch_size / 2),
                shuffle=True) for x in ['source', 'd_source', 'test_source', 'target', 'target_test', 'd_pseudo_source',\
                                        'source_depth', 'd_source_depth', 'test_source_depth', 'target_depth_gen', 'target_test_depth', 'd_pseudo_source_depth']}
# , 'test'
dset_loaders['test'] = torch.utils.data.DataLoader(dsets['test'], batch_size=1,
                shuffle=False)
dset_loaders['test_depth'] = torch.utils.data.DataLoader(dsets['test'], batch_size=1,
                shuffle=False)

dset_loaders['depth_gen_draw'] = torch.utils.data.DataLoader(dsets['depth_gen_draw'], batch_size=1,
                shuffle=False)

source_batches_per_epoch = np.floor(len(dsets['source']) * 2 / args.batch_size).astype(np.int16)
target_batches_per_epoch = np.floor(len(dsets['target']) * 2 / args.batch_size).astype(np.int16)

pre_epochs = int(args.pretrain_sample / len(dsets['source'])) + CLS_EP
total_epochs = int(pre_epochs + args.train_sample / len(dsets['source'])) 

# pre_epochs = int(args.pretrain_sample / min(len(dsets['source']), len(dsets['target'])))
# total_epochs = int(pre_epochs + args.train_sample / min(len(dsets['source']), len(dsets['target'])))

pre_iters = int(args.pretrain_sample * 2 / args.batch_size)
total_iters = int(pre_iters + args.train_sample * 2 / args.batch_size)

if args.useLargeLREpoch:
    lr_epochs = 10000
else:
    lr_epochs = total_epochs

######################################################################
# Finetuning the convnet
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
                'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
                'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',}
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# test_source
#pseudo_dataset: dsets['target_test']
def cal_pseudo_label_set(model, pseudo_dataset):

    dset_loaders[pseudo_dataset] = torch.utils.data.DataLoader(dsets[pseudo_dataset], batch_size=1, shuffle=False)
            
    Pseudo_set_f = []
    pseudo_counter = 0

    for pseudo_inputs, pseudo_labels, pseudo_path, _ in dset_loaders[pseudo_dataset]:

        pseudo_inputs = Variable(pseudo_inputs.cuda())

        domain_labels_t = Variable(torch.FloatTensor([0.]*len(pseudo_inputs)).cuda())

        ini_weight = Variable(torch.FloatTensor([1.0]*len(pseudo_inputs)).cuda())

        class_t, domain_out_t, confid_rate = model('pseudo_discriminator', pseudo_inputs,[],[],domain_labels_t,ini_weight)

        # prediction confidence weight

        # domain variance

        dom_prob = F.sigmoid(domain_out_t.squeeze())

        top_prob, top_label = torch.topk(F.softmax(class_t.squeeze()), 1)

        # dom_confid = 1 - dom_prob.data[0]

        # s_tuple = (pseudo_path, top_label.data[0], confid_rate.data[0], dom_prob.data[0], confid_rate.data[0], int(pseudo_labels[0]))
        s_tuple = (pseudo_path, top_label.data[0], confid_rate.data[0], dom_prob.data[0], confid_rate.data[0], int(pseudo_labels[0]))
        Pseudo_set_f.append(s_tuple)
        # -------- sort domain variance score, reduce memory ------------
        fake_sample = int(int(top_label[0].cpu().data[0]) != int(pseudo_labels[0]))

        # total_pseudo_errors += fake_sample
        pseudo_counter += 1

    return Pseudo_set_f

# test_source
def cal_test_source_accuracy(model, test_source_dataset):

    test_source_corrects = 0
    dset_loaders[test_source_dataset] = torch.utils.data.DataLoader(dsets[test_source_dataset], batch_size=1, shuffle=False)
        
    for test_source_input, test_source_label in dset_loaders[test_source_dataset]:

        test_source_input, test_source_label = Variable(test_source_input.cuda()), Variable(test_source_label.cuda())
        test_source_outputs = model('test', test_source_input)

        # ------------ test classification statistics ------------
        _, test_source_preds = torch.max(test_source_outputs.data, 1)
        test_source_corrects += torch.sum(test_source_preds == test_source_label.data)

    # epoch_loss = epoch_loss / len(dsets['test'])
    #             epoch_acc = epoch_corrects / len(dsets['test'])
    #             epoch_acc_t = epoch_acc

    acc_test_source = test_source_corrects / len(dsets[test_source_dataset])

    return acc_test_source

def train_model(model, model2, model3, optimizer, dom_optimizer, dom_feat_optimizer, optimizer2, dom_optimizer2, dom_feat_optimizer2, depth_gen_optimizer, depth_disc_optimizer, depth_net_optimizer, depth_enc_optimizer, cls_lr_scheduler, dom_w_lr_scheduler, feature_params, num_epochs=500, gen_epochs=100):
    since = time.time()

    optimizer_list = [optimizer, dom_optimizer, dom_feat_optimizer,optimizer2, dom_optimizer2, dom_feat_optimizer2, depth_gen_optimizer, depth_disc_optimizer, depth_net_optimizer, depth_enc_optimizer]

    # ----- initialise variables ----

    double_desc = False
    double_desc2 = False

    best_model = model
    best_model2 = model2
    best_acc = 0.0
    epoch_acc_s = 0.0
    epoch_acc_s2 = 0
    epoch_acc_t = 0.0
    epoch_loss_s = 0.0
    pre_epoch_acc_s = 0.0
    pre_epoch_loss_s = 0.0
    total_epoch_acc_s = 0.0
    total_epoch_loss_s = 0.0

    avg_epoch_acc_s = 0.0

    pre_epoch_acc_s2 = 0.0
    pre_epoch_loss_s2 = 0.0
    total_epoch_acc_s2 = 0.0
    total_epoch_loss_s2 = 0.0

    prev_epoch_acc_s = 0.0
    prev_epoch_acc_s2 = 0.0

    avg_epoch_acc_s2 = 0.0

    avg_epoch_loss_s = 0.0
    total_threshold = 0.0
    avg_threshold = 0.1
    epoch_lr_mult = 0.0
    threshold_count = 0
    threshold_count2 = 0
    threshold_list = []

    test_source_count = 0
    avg_test_source_acc = 0.0
    total_test_source_acc = 0.0

    source_step_count = 0
    target_step_count = 0

    iters = 0
    current_iters = 0

    epoch_point = []
    class_loss_point = []
    domain_loss_point = []

    source_acc_point = []
    source_acc_point2 = []
    domain_acc_point = []

    target_loss_point = []
    target_acc_point = []
    target_acc_point2 = []
    target_acc_point_co = []

    set_len_point = []

    confid_threshold_point = []

    epoch_point = []
    lr_point = []

    domain_loss_point_l1 = []
    domain_loss_point_l2 = []
    domain_loss_point_l3 = []

    domain_acc_point_l1 = []
    domain_acc_point_l2 = []
    domain_acc_point_l3 = []

    # pseudo_source_dict = [(i,j,k) for (i,j,k) in dset_loaders['pseudo_source']]

    # for inputs, labels in dset_loaders['train']:
    #     inputs_var, labels_var = Variable(inputs.cuda()), Variable(labels.cuda())
    #     inputs_list, labels_list = inputs.numpy(), labels.numpy().tolist()
    #     for i in range(len(inputs_list)):
    #         Source_set.append((torch.from_numpy(np.array(inputs_list[i])), labels_list[i]))
    try:
        os.makedirs('./gen/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    try:
        os.makedirs('./gen/%s2%s/'%(args.source_set, args.target_set))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # ------------------------- two model train ----------------------------------

    for epoch in range(lr_epochs):

        loss_meter = AverageMeter()
        print('Epoch {}/{}/{}'.format(epoch, num_epochs - 1, lr_epochs))
        print('-' * 10)
        draw_flag = False
        if epoch >= CLS_EP:
            epoch_point.append(str(epoch))

        # ----------------------------------------------------------
        # --------------- Training and Testing Phase ---------------
        # ----------------------------------------------------------

        for phase in ['train', 'test']:

            # ----- initialise common variables -----
            epoch_loss = 0.0
            epoch_loss2 = 0.0
            epoch_corrects = 0
            epoch_corrects2 = 0
            epoch_corrects_co = 0
            
            test_corrects = {}
            test_totals = {}
            # ----------------------------------------------------------
            # ------------------ Training Phase ------------------------
            # ----------------------------------------------------------

            if phase == 'train':

                # ----- initialise common variables -----

                domain_epoch_loss = 0.0
                total_epoch_loss = 0.0
                domain_epoch_corrects = 0
                epoch_discrim_bias = 0.0


                gen_epoch_loss_gen = 0.0
                domain_epoch_loss_gen = 0.0
                domain_epoch_corrects_gen = 0

                #######################
                ###### modality 1 #####
                #######################

                # ---- classifier -----
                source_pointer = 0
                pseudo_pointer = 0

                total_source_pointer = 0
                total_pseudo_pointer = 0

                source_pointer2 = 0
                pseudo_pointer2 = 0

                total_source_pointer2 = 0
                total_pseudo_pointer2 = 0

                # ---- domain discriminator -----

                # ------- source part --------
                d_source_pointer = 0
                d_source_feat_pointer = 0

                d_pseudo_source_pointer = 0
                d_pseudo_source_feat_pointer = 0

                total_d_source_pointer = 0
                total_d_source_feat_pointer = 0

                total_d_pseudo_source_pointer = 0
                total_d_pseudo_source_feat_pointer = 0

                # ------- target part --------
                d_target_pointer = 0
                d_target_feat_pointer = 0

                d_pseudo_pointer = 0
                d_pseudo_feat_pointer = 0
                
                d_pseudo_all_pointer = 0
                d_pseudo_all_feat_pointer = 0

                total_d_target_pointer = 0
                total_d_target_feat_pointer = 0

                total_d_pseudo_pointer = 0
                total_d_pseudo_feat_pointer = 0

                total_d_pseudo_all_pointer = 0
                total_d_pseudo_all_feat_pointer = 0

                # --------------------------------
                depth_gen_pointer = 0
                total_depth_gen_pointer = 0

                depth_gen_pointer2 = 0
                total_depth_gen_pointer2 = 0

                # ----------------------------
                batch_count = 0
                class_count = 0
                domain_counts = 0
                domain_counts_gen = 0
                
                # -----------------------------
                # ------- modality 2 -------
                # -------------------------

                d_source_pointer2 = 0
                d_source_feat_pointer2 = 0

                d_pseudo_source_pointer2 = 0
                d_pseudo_source_feat_pointer2 = 0

                total_d_source_pointer2 = 0
                total_d_source_feat_pointer2 = 0

                total_d_pseudo_source_pointer2 = 0
                total_d_pseudo_source_feat_pointer2 = 0

                # ------- target part --------
                d_target_pointer2 = 0
                d_target_feat_pointer2 = 0

                d_pseudo_pointer2 = 0
                d_pseudo_feat_pointer2 = 0
                
                d_pseudo_all_pointer2 = 0
                d_pseudo_all_feat_pointer2 = 0

                total_d_target_pointer2 = 0
                total_d_target_feat_pointer2 = 0

                total_d_pseudo_pointer2 = 0
                total_d_pseudo_feat_pointer2 = 0

                total_d_pseudo_all_pointer2 = 0
                total_d_pseudo_all_feat_pointer2 = 0

                # ----------------------------

                domain_epoch_corrects2 = 0
                total_epoch_loss2 = 0

                # ----------------------------

                batch_count2 = 0
                class_count2 = 0
                domain_counts2 = 0

                confid_threshold2 = 0

                # total_iters = 0

                domain_epoch_loss_l1 = 0.0
                domain_epoch_loss_l2 = 0.0
                domain_epoch_loss_l3 = 0.0

                domain_epoch_corrects_l1 = 0
                domain_epoch_corrects_l2 = 0
                domain_epoch_corrects_l3 = 0

                w_main = 0.0 
                w_l1 = 0.0 
                w_l2 = 0.0 
                w_l3 = 0.0 

                confid_threshold = 0

                # -----------------------------------------------
                # ----------------- Pre-train -------------------
                # -----------------------------------------------

                if epoch >= pre_epochs:
                    
                    # -----------------------------------------------
                    # ------------ Pseudo Labelling -----------------
                    # -----------------------------------------------

                    # ------- with iterative pseudo sample filter --------

                    model.train(False)
                    model.eval()
                    model2.train(False)
                    model2.eval()
                    model3.train(False)
                    model3.eval()

                    Pseudo_set = []
                    Select_T1_set = []
                    total_pseudo_errors = 0

                    Pseudo_set2 = []
                    Select_T1_set2 = []
                    total_pseudo_errors2 = 0

                    Pseudo_set_co = []
                    Select_T1_set_co = []
                    total_pseudo_errors_co = 0


                    dset_loaders['target_test_total'] = torch.utils.data.DataLoader(dsets['target_test_total'],
                                                        batch_size=1, shuffle=False)
            
                    pseudo_counter = 0
            
                    for test_inputs, test_inputs2, test_labels, test_path, test_path2, _ in dset_loaders['target_test_total']:

                        test_inputs_var1 = Variable(test_inputs, volatile=True).cuda()
                        test_inputs2 = Variable(test_inputs2, volatile=True).cuda()

                        # domain_labels_t = Variable(torch.FloatTensor([0.]*len(test_inputs)).cuda())
                        # domain_labels_t_d = Variable(torch.FloatTensor([0.]*len(test_inputs_d)).cuda())

                        # ini_weight = Variable(torch.FloatTensor([1.0]*len(test_inputs)).cuda())

                        output, domain_out = model('pseudo_discriminator', test_inputs_var1)
                        dom_prob = F.sigmoid(domain_out.squeeze()).mean()
                        class_t_avg_prob = F.softmax(output, dim=1)
                        top_prob, top_label = torch.topk(class_t_avg_prob.squeeze(), 1)

                        if not DEPTH_GT:
                            test_inputs_var2 = Variable(test_inputs, volatile=True).cuda()
                            
                            if USE_Conditional:
                                # use predicted pseudo label from rgb stream
                                t_np_labels = top_label.data.cpu().numpy()
                                t_onehot_np = np.zeros((t_np_labels.size, args.num_class))
                                t_onehot_np[np.arange(t_np_labels.size), t_np_labels] = 1
                                t_classonehot_var = Variable(torch.FloatTensor(t_onehot_np),volatile=True).cuda()
                                test_pred_depth = model3(test_inputs_var2, train_conditional_gen=True, classonehot=t_classonehot_var)
                                # test_depth_var = Variable(test_pred_depth.data).cuda()

                            else:
                                test_pred_depth = model3(test_inputs_var2, train_gen=True)
                                # test_depth_var = Variable(test_pred_depth.data).cuda()
                            output2, domain_out2 = model2('pseudo_discriminator', test_pred_depth)
                        else:
                            output2, domain_out2 = model2('pseudo_discriminator', test_inputs2)

                        # ----- pseudo labels -----------

                        dom_prob2 = F.sigmoid(domain_out2.squeeze()).mean()
                        dom_prob_co = (RGB_Ratio * dom_prob.data[0] + Depth_Ratio * dom_prob2.data[0]) * (1.0 / (RGB_Ratio + Depth_Ratio))

                        class_t_avg_prob2 = F.softmax(output2, dim=1)
                        class_t_avg_prob_co = (RGB_Ratio * class_t_avg_prob.cpu() + Depth_Ratio * class_t_avg_prob2.cpu()) * (1.0 / (RGB_Ratio + Depth_Ratio))


                        top_prob2, top_label2 = torch.topk(class_t_avg_prob2.squeeze(), 1)
                        top_prob_co, top_label_co = torch.topk(class_t_avg_prob_co.squeeze(), 1)

                        confid_rate = top_prob
                        confid_rate2 = top_prob2
                        confid_rate_co = top_prob_co

                        s_tuple = (test_path, test_path2, top_label.data[0], confid_rate.data[0], dom_prob.data[0], confid_rate.data[0], int(test_labels[0]))
                        s_tuple2 = (test_path, test_path2, top_label2.data[0], confid_rate2.data[0], dom_prob2.data[0], confid_rate2.data[0], int(test_labels[0]))
                        s_tuple_co = (test_path, test_path2, top_label_co.data[0], confid_rate_co.data[0], dom_prob_co, confid_rate_co.data[0], int(test_labels[0]))

                        Pseudo_set.append(s_tuple)
                        Pseudo_set2.append(s_tuple2)
                        Pseudo_set_co.append(s_tuple_co)

                        # -------- sort domain variance score, reduce memory ------------
                        fake_sample = int(int(top_label[0].cpu().data[0]) != int(test_labels[0]))
                        fake_sample2 = int(int(top_label2[0].cpu().data[0]) != int(test_labels[0]))
                        fake_sample_co = int(int(top_label_co[0].cpu().data[0]) != int(test_labels[0]))

                        total_pseudo_errors += fake_sample
                        total_pseudo_errors2 += fake_sample2
                        total_pseudo_errors_co += fake_sample_co
                        pseudo_counter += 1

                    print("total_pseudo_errors, 2, co, total", total_pseudo_errors, total_pseudo_errors2, total_pseudo_errors_co, total_pseudo_errors, len(dsets['target_test_total']))
                    # ----------------- calculate pseudo threshold -------------------
                    threshold_count += 1
                    threshold_count2 += 1

                    #########################################
                    # self -------------------------
                    if args.ReTestSource:
                        current_test_source_acc = cal_test_source_accuracy(model, 'test_source')
                        max_threshold = args.totalPseudoChange
                    else:
                        current_test_source_acc = epoch_acc_s
                        current_test_source_acc2 = epoch_acc_s2
                        if args.usePrevAcc:
                            avg_test_source_acc = prev_epoch_acc_s
                            avg_test_source_acc2 = prev_epoch_acc_s2
                        else:
                            avg_test_source_acc = avg_epoch_acc_s
                            avg_test_source_acc2 = avg_epoch_acc_s2
                        max_threshold = args.epochs
                        max_threshold2 = args.epochs2

                    if current_test_source_acc < avg_test_source_acc:
                        if not double_desc:
                            double_desc = True

                            if args.usingTriDec:
                                threshold_count -= 2
                            elif args.usingDoubleDecrease:
                                threshold_count -= 1
                        else:
                            if args.usingTriDec:
                                threshold_count -= 3
                            else:
                                threshold_count -= 2
                    else:
                        double_desc = False

                    if current_test_source_acc2 < avg_test_source_acc2:
                        if not double_desc2:
                            double_desc2 = True

                            if args.usingTriDec2:
                                threshold_count2 -= 2
                            elif args.usingDoubleDecrease2:
                                threshold_count2 -= 1
                        else:
                            if args.usingTriDec2:
                                threshold_count2 -= 3
                            else:
                                threshold_count2 -= 2
                    else:
                        double_desc2 = False

                    ########################################
                    # --------------- fuse -----------------

                    # if args.ReTestSource:
                    #     current_test_source_acc = cal_test_source_accuracy(model, 'test_source')
                    #     max_threshold = args.totalPseudoChange
                    # else:
                    #     current_test_source_acc = epoch_acc_s
                    #     current_test_source_acc2 = epoch_acc_s2
                    #     if args.usePrevAcc:
                    #         avg_test_source_acc = prev_epoch_acc_s
                    #         avg_test_source_acc2 = prev_epoch_acc_s2
                    #     else:
                    #         avg_test_source_acc = avg_epoch_acc_s
                    #         avg_test_source_acc2 = avg_epoch_acc_s2
                    #     max_threshold = args.epochs
                    #     max_threshold2 = args.epochs2

                    # if current_test_source_acc < avg_test_source_acc:
                    #     if not double_desc:
                    #         double_desc = True

                    #         if args.usingTriDec:
                    #             threshold_count -= 2
                    #         elif args.usingDoubleDecrease:
                    #             threshold_count -= 1
                    #     else:
                    #         if args.usingTriDec:
                    #             threshold_count -= 3
                    #         else:
                    #             threshold_count -= 2
                    # else:
                    #     double_desc = False

                    # if current_test_source_acc2 < avg_test_source_acc2:
                    #     if not double_desc2:
                    #         double_desc2 = True

                    #         if args.usingTriDec2:
                    #             threshold_count2 -= 2
                    #         elif args.usingDoubleDecrease2:
                    #             threshold_count2 -= 1
                    #     else:
                    #         if args.usingTriDec2:
                    #             threshold_count2 -= 3
                    #         else:
                    #             threshold_count2 -= 2
                    # else:
                    #     double_desc2 = False

                    print("current_test_source_acc", current_test_source_acc, "avg_test_source_acc", avg_test_source_acc, "prev_epoch_acc_s",prev_epoch_acc_s, "double_desc", double_desc)

                    print("current_test_source_acc2", current_test_source_acc2, "avg_test_source_acc2", avg_test_source_acc2, "prev_epoch_acc_s2",prev_epoch_acc_s2, "double_desc2", double_desc2)        

                    # sourceTestIter
                    # totalPseudoChange

                    pseu_n = args.defaultPseudoRatio + threshold_count / max_threshold
                    pseu_n = pull_back_to_one(pseu_n)

                    print("threshold_count: {} pseudo_ratio:{:.4f}".format(threshold_count, pseu_n))
                    
                    pseu_n2 = args.defaultPseudoRatio + threshold_count2 / max_threshold2
                    pseu_n2 = pull_back_to_one(pseu_n2)

                    print("threshold_count: {} pseudo_ratio:{:.4f}".format(threshold_count2, pseu_n2))

                    # ----------------- calculate avg test source acc -------------------
                    # if args.ReTestSource:
                    #     test_source_count += 1
                    #     total_test_source_acc += current_test_source_acc
                    #     avg_test_source_acc = total_test_source_acc / test_source_count

                    # ----------------- sort pseudo datasets -------------------

                    pseudo_num = len(Pseudo_set)
                    Sorted_confid_set = sorted(Pseudo_set, key=lambda tup: tup[3], reverse=True)

                    pseudo_num2 = len(Pseudo_set2)
                    Sorted_confid_set2 = sorted(Pseudo_set2, key=lambda tup: tup[3], reverse=True)

                    pseudo_num_co = len(Pseudo_set_co)
                    Sorted_confid_set_co = sorted(Pseudo_set_co, key=lambda tup: tup[3], reverse=True)
                    # ----------------- get pseudo_C set -------------------

                    t_p = pseu_n * RGB_Ratio

                    if USE_COFUSE:

                        confid_pseudo_num_co = int(t_p*pseudo_num_co)
                        
                        Select_confid_set_co = Sorted_confid_set_co[:confid_pseudo_num_co]


                        if len(Select_confid_set_co) > 0:
                            confid_threshold = Select_confid_set_co[-1][3]

                        Select_T1_set = Select_confid_set_co

                        if USE_Half:
                            confid_pseudo_num_h = int(t_p*pseudo_num)

                            Select_confid_set1 = Sorted_confid_set[:confid_pseudo_num_h]
                            
                            Select_T1_set_1only = Select_confid_set1

                    else:
                        confid_pseudo_num = int(t_p*pseudo_num)

                        Select_confid_set = Sorted_confid_set[:confid_pseudo_num]


                        if len(Select_confid_set) > 0:
                            confid_threshold = Select_confid_set[-1][3]

                        Select_T1_set = Select_confid_set

                    # ----------------- get pseudo_C set 2 -------------------
        
                    t_p2 = pseu_n2 * Depth_Ratio

                    if USE_COFUSE:

                        confid_pseudo_num_co2 = int(t_p2*pseudo_num_co)
                        
                        Select_confid_set_co2 = Sorted_confid_set_co[:confid_pseudo_num_co2]


                        if len(Select_confid_set_co2) > 0:
                            confid_threshold2 = Select_confid_set_co2[-1][3]

                        Select_T1_set2 = Select_confid_set_co2

                        if USE_INDI:
                            Select_T1_set2_only = Sorted_confid_set2[:confid_pseudo_num_co2]

                    else:

                        confid_pseudo_num2 = int(t_p2*pseudo_num2)
                        Select_confid_set2 = Sorted_confid_set2[:confid_pseudo_num2]

                        if len(Select_confid_set2) > 0:
                            confid_threshold2 = Select_confid_set2[-1][3]

                        Select_T1_set2 = Select_confid_set2

                    ####################################################

                    if USE_COFUSE:

                        Select_T1_set = Select_T1_set + Select_T1_set2
                        # Select_T2_set = Select_T2_set + Select_T2_set2
                        # Select_T1T2_set = Select_T1T2_set + Select_T1T2_set2

                        Select_T1_set2 = Select_T1_set

                        if USE_INDI:
                            Select_T1_set2 = Select_T1_set2 + Select_T1_set2_only 
                        # Select_T2_set2 = Select_T2_set
                        # Select_T1T2_set2 = Select_T1T2_set

                    print("lenT1", len(Select_T1_set), len(Select_T1_set2))
                    # ----------------- get pseudo datasets -------------------
                    if USE_Half:
                        if len(Select_T1_set_1only) > 0:
                            dsets['pseudo'] = imagefolder.TotalPathFolder(Select_T1_set_1only, data_transforms['target'], data_transforms['depth'])
                        if len(Select_T1_set2) > 0:
                            dsets['pseudo_depth'] = imagefolder.TotalPathFolder(Select_T1_set2, data_transforms['target'], data_transforms['depth'])

                        if len(Select_T1_set_1only) > 0:
                            dsets['d_pseudo_all'] = imagefolder.TotalPathFolder(Select_T1_set_1only, data_transforms['target'], data_transforms['depth'])
                            dsets['d_pseudo_all_feat'] = imagefolder.TotalPathFolder(Select_T1_set_1only, data_transforms['target'], data_transforms['depth'])

                        if len(Select_T1_set2) > 0:
                            dsets['d_pseudo_all_depth'] = imagefolder.TotalPathFolder(Select_T1_set2, data_transforms['target'], data_transforms['depth'])
                            dsets['d_pseudo_all_feat_depth'] = imagefolder.TotalPathFolder(Select_T1_set2, data_transforms['target'], data_transforms['depth'])
                    else:

                        if len(Select_T1_set) > 0:
                            dsets['pseudo'] = imagefolder.TotalPathFolder(Select_T1_set, data_transforms['target'], data_transforms['depth'])
                        if len(Select_T1_set2) > 0:
                            dsets['pseudo_depth'] = imagefolder.TotalPathFolder(Select_T1_set2, data_transforms['target'], data_transforms['depth'])

                        if len(Select_T1_set) > 0:
                            dsets['d_pseudo_all'] = imagefolder.TotalPathFolder(Select_T1_set, data_transforms['target'], data_transforms['depth'])
                            dsets['d_pseudo_all_feat'] = imagefolder.TotalPathFolder(Select_T1_set, data_transforms['target'], data_transforms['depth'])

                        if len(Select_T1_set2) > 0:
                            dsets['d_pseudo_all_depth'] = imagefolder.TotalPathFolder(Select_T1_set2, data_transforms['target'], data_transforms['depth'])
                            dsets['d_pseudo_all_feat_depth'] = imagefolder.TotalPathFolder(Select_T1_set2, data_transforms['target'], data_transforms['depth'])

                    # ----------------- reload pseudo set ---------------------------

                    confid_threshold_point.append(float("%.4f" % confid_threshold))

                # ---------------------------------------------------------
                # -------- Pseudo + Source Classifier Training ------------
                # ---------------------------------------------------------

                # -------- loop through source dataset ---------

                model.train(True) # Set model to training mode
                model2.train(True) # Set model to training mode

                for param in model.parameters():
                    param.requires_grad = True

                for param in model.disc_activate.parameters():
                    param.requires_grad = False

                # if epoch < pre_epochs:
                #     for param in model.disc_weight.parameters():
                #         param.requires_grad = False

                for param in model2.parameters():
                    param.requires_grad = True

                for param in model2.disc_activate.parameters():
                    param.requires_grad = False

                # if epoch < pre_epochs:
                #     for param in model2.disc_weight.parameters():
                #         param.requires_grad = False

                # ini_w_main = Variable(torch.FloatTensor([float(args.main_w)]).cuda())
                # ini_w_l1 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda())
                # ini_w_l2 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda())
                # ini_w_l3 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda())

                # ini_w_main_2 = Variable(torch.FloatTensor([float(args.main_w2)]).cuda())
                # ini_w_l1_2 = Variable(torch.FloatTensor([float(args.form_w2/3)]).cuda())
                # ini_w_l2_2 = Variable(torch.FloatTensor([float(args.form_w2/3)]).cuda())
                # ini_w_l3_2 = Variable(torch.FloatTensor([float(args.form_w2/3)]).cuda())

                # ---------------------------------------------------------------------
                # ------- Source + Pseudo + Target Dataset Preparation ---------
                # ---------------------------------------------------------------------

                source_size = len(dsets['source'])
                pseudo_size = len(dsets['pseudo'])
                d_target_size = len(dsets['target'])
                d_pseudo_all_size = len(dsets['d_pseudo_all'])

                total_errors = 0
                for ss in dsets['pseudo']:
                    fake_sample = int(ss[2] != ss[-1])
                    total_errors += fake_sample 

                print("Select error/total selected = {}/{}".format(total_errors,pseudo_size)) 


                if pseudo_size == 0:
                    source_batchsize = int(args.batch_size / 2) 
                    pseudo_batchsize = 0
                    d_source_batchsize = int(args.batch_size / 2)
                    d_target_batchsize = int(args.batch_size / 2)
                    d_pseudo_source_batchsize = 0
                    d_pseudo_all_batchsize = 0
                else:
                    # source_batchsize = int(round(float(args.batch_size / 2) * source_size / float(source_size + pseudo_size)))
                    source_batchsize = int(int(args.batch_size / 2) * source_size / (source_size + pseudo_size))

                    if source_batchsize < int(int(args.batch_size / 2) / 2):
                        source_batchsize = int(int(args.batch_size / 2) / 2)
                    if source_batchsize == int(args.batch_size / 2):
                        source_batchsize -= 1

                    pseudo_batchsize = int(args.batch_size / 2) - source_batchsize
                
                    d_source_batchsize = source_batchsize
                    d_pseudo_source_batchsize = pseudo_batchsize

                    d_pseudo_all_batchsize = 0
                    if d_pseudo_all_size > 0:
                        d_pseudo_all_batchsize = int(round(int(args.batch_size / 2) * d_pseudo_all_size / d_target_size))

                        if d_pseudo_all_batchsize == 0:
                            d_pseudo_all_batchsize = 1

                        if d_pseudo_all_batchsize >= int(args.batch_size / 2):
                            d_pseudo_all_batchsize = int(args.batch_size / 2) - 1


                    d_target_batchsize = int(args.batch_size / 2) - d_pseudo_all_batchsize

                    pseudo_iter = iterator_reset('pseudo', 'pseudo', pseudo_batchsize)

                    d_pseudo_source_iter = iterator_reset('d_pseudo_source', 'd_pseudo_source', d_pseudo_source_batchsize)
                    d_pseudo_source_feat_iter = iterator_reset('d_pseudo_source_feat', 'd_pseudo_source', d_pseudo_source_batchsize)

                    if d_pseudo_all_size > 0:

                        d_pseudo_all_iter = iterator_reset('d_pseudo_all', 'd_pseudo_all', d_pseudo_all_batchsize)
                        d_pseudo_all_feat_iter = iterator_reset('d_pseudo_all_feat', 'd_pseudo_all_feat', d_pseudo_all_batchsize)


                print("batchsizes: S, Pseu, d_all, dt", source_batchsize, pseudo_batchsize, d_pseudo_all_batchsize, d_target_batchsize)

                source_iter = iterator_reset('source','source',source_batchsize)

                d_source_iter = iterator_reset('d_source','source',d_source_batchsize)
                d_source_feat_iter = iterator_reset('d_source_feat','source',d_source_batchsize)

                d_target_iter = iterator_reset('d_target','target',d_target_batchsize)
                d_target_feat_iter = iterator_reset('d_target_feat','target',d_target_batchsize)

                # ------------------------------------------------------------------
                # -------- reset source and pseudo batch ratio for modality 2-------

                source_size2 = len(dsets['source_depth'])
                pseudo_size2 = len(dsets['pseudo_depth'])
                d_target_size2 = len(dsets['target_depth_gen']) * (RGB_Ratio+Depth_Ratio)
                d_pseudo_all_size2 = len(dsets['d_pseudo_all_depth'])

                # pseudo_size2 = int(round(pseudo_size2 / 2))
                # d_target_size2 = int(round(d_target_size2 / 2))
                # d_pseudo_all_size2 = int(round(d_pseudo_all_size2 / 2))

                total_errors2 = 0
                for ss in dsets['pseudo_depth']:
                    fake_sample = int(ss[2] != ss[-1])
                    total_errors2 += fake_sample 

                print("Select2 error/total selected2 = {}/{}".format(total_errors2, pseudo_size2)) 

                # print("pseudo_all", [i[0] for i in dsets['d_pseudo_all']])

                if pseudo_size2 == 0:
                    source_batchsize2 = int(args.batch_size / 2) 
                    pseudo_batchsize2 = 0
                    d_source_batchsize2 = int(args.batch_size / 2)
                    d_target_batchsize2 = int(args.batch_size / 2)
                    d_pseudo_source_batchsize2 = 0
                    d_pseudo_all_batchsize2 = 0
                else:
                    # source_batchsize = int(round(float(args.batch_size / 2) * source_size / float(source_size + pseudo_size)))
                    source_batchsize2 = int(int(args.batch_size / 2) * source_size2 / (source_size2 + pseudo_size2))

                    if source_batchsize2 < int(int(args.batch_size / 2) / 2):
                        source_batchsize2 = int(int(args.batch_size / 2) / 2)
                    if source_batchsize2 == int(args.batch_size / 2):
                        source_batchsize2 -= 1

                    pseudo_batchsize2 = int(args.batch_size / 2) - source_batchsize2
                
                    d_source_batchsize2 = source_batchsize2
                    d_pseudo_source_batchsize2 = pseudo_batchsize2

                    d_pseudo_all_batchsize2 = 0
                    if d_pseudo_all_size2 > 0:
                        d_pseudo_all_batchsize2 = int(round(int(args.batch_size / 2) * d_pseudo_all_size2 / d_target_size2))

                        if d_pseudo_all_batchsize2 == 0:
                            d_pseudo_all_batchsize2 = 1

                        if d_pseudo_all_batchsize2 >= int(args.batch_size / 2):
                            d_pseudo_all_batchsize2 = int(args.batch_size / 2) - 1

                    d_target_batchsize2 = int(args.batch_size / 2) - d_pseudo_all_batchsize2

                    pseudo_iter2 = iterator_reset('pseudo_depth', 'pseudo_depth', pseudo_batchsize2)

                    d_pseudo_source_iter2 = iterator_reset('d_pseudo_source_depth', 'source_depth', d_pseudo_source_batchsize2)
                    d_pseudo_source_feat_iter2 = iterator_reset('d_pseudo_source_feat_depth', 'source_depth', d_pseudo_source_batchsize2)

                    if d_pseudo_all_size2 > 0:

                        d_pseudo_all_iter2 = iterator_reset('d_pseudo_all_depth', 'd_pseudo_all_depth', d_pseudo_all_batchsize2)
                        d_pseudo_all_feat_iter2 = iterator_reset('d_pseudo_all_feat_depth', 'd_pseudo_all_feat_depth', d_pseudo_all_batchsize2)


                source_iter2 = iterator_reset('source_depth','source_depth', source_batchsize2)

                d_source_iter2 = iterator_reset('d_source_depth','source_depth', d_source_batchsize2)
                d_source_feat_iter2 = iterator_reset('d_source_feat_depth','source_depth', d_source_batchsize2)

                d_target_iter2 = iterator_reset('d_target_depth_gen','target_depth_gen', d_target_batchsize2)
                d_target_feat_iter2 = iterator_reset('d_target_feat_depth_gen','target_depth_gen', d_target_batchsize2)
                ###################################################

                depth_gen_iter = iterator_reset('depth_gen','depth_gen',int(args.batch_size/ 2))

                depth_gen_iter2 = iterator_reset('depth_gen_target','depth_gen_target',int(args.batch_size/ 2))

                # -----------------------------------------------------------------

                if epoch < CLS_EP:
                    model.train(False) # Set model to training mode
                    model2.train(False) # Set model to training mode
                    model3.train(True)
                
                    for param in model.parameters():
                        param.requires_grad = False

                    for param in model2.parameters():
                        param.requires_grad = False

                    for p in model3.parameters():
                        p.requires_grad = True


                    while depth_gen_pointer < len(depth_gen_iter):

                        p_gen = epoch / CLS_EP

                        p_gen = pull_back_to_one(p_gen)
                        l_gen = (2. / (1. + np.exp(-10. * p_gen))) - 1

                        lr_mult_gen = (1. + 10 * p_gen)**(-0.75)

                        depth_gen_optimizer, _ = cls_lr_scheduler(depth_gen_optimizer, lr_mult_gen)
                        depth_disc_optimizer, _ = cls_lr_scheduler(depth_disc_optimizer, lr_mult_gen)
                        depth_enc_optimizer, _ = cls_lr_scheduler(depth_enc_optimizer, lr_mult_gen)

                        # make sure to skip the last batch if the batch length is not enough(drop last)
                        batch_count += 1
                        if (batch_count * int(args.batch_size / 2) > len(dsets['depth_gen'])):
                            continue
                        # --------------------- ------------------------- -----------------------
                        # --------------------- classification part batch -----------------------
                        # --------------------- ------------------------- -----------------------
                        
                        depth_gen_iter, inputs_gen, depth_inputs_gen, labels, depth_gen_pointer, total_depth_gen_pointer \
                                        = iterator_update(depth_gen_iter, 'depth_gen', int(args.batch_size / 2), 
                                                          depth_gen_pointer, total_depth_gen_pointer, "depth_gen")


                        depth_gen_iter2, inputs2_gen, depth_inputs2_gen, labels2, depth_gen_pointer2, total_depth_gen_pointer2 \
                                        = iterator_update(depth_gen_iter2, 'depth_gen_target', int(args.batch_size / 2), 
                                                          depth_gen_pointer2, total_depth_gen_pointer2, "depth_gen")

                        #################### Train Recon #########################


                        rgb = Variable(inputs_gen.cuda())
                        depth = Variable(depth_inputs_gen.cuda())

                        if USE_Conditional:
                            # learn with source depth and label
                            np_labels = labels.numpy()
                            onehot_np = np.zeros((np_labels.size, args.num_class))
                            onehot_np[np.arange(np_labels.size), np_labels] = 1
                            classonehot_var = Variable(torch.FloatTensor(onehot_np)).cuda()
                            pred_depth = model3(rgb, train_conditional_gen=True, classonehot=classonehot_var)
                        # print(rgb_combine)

                        pred_depth = model3(rgb, train_gen=True)

                        model3.zero_grad()

                        criterionCAE = nn.L1Loss()
                        genloss = criterionCAE(pred_depth, depth)
                        genloss.backward()

                        gen_epoch_loss_gen += genloss.data[0]

                        depth_gen_optimizer.step()

                        print("Epoch:",epoch, " Losses:",genloss[0].data[0])
                        # ###################################################
                        if epoch >= POS_CLS_EP:
                            # rev learn with source target & domain label
                            rgb_combine = Variable(torch.cat((inputs_gen,inputs2_gen),0)).cuda()
                            
                            if epoch == POS_CLS_EP:
                                l = 0.3

                            if USE_Conditional:
                                domain_outputs_gen = model3(rgb_combine, train_conditional_dom=True, l=l_gen)
                            else:
                                domain_outputs_gen = model3(rgb_combine, train_dom=True, l=l_gen)

                            domain_labels_gen = Variable(torch.FloatTensor([1.]*int(args.batch_size / 2) 
                                                      +[0.]*int(args.batch_size / 2)).cuda())
                            
                            domain_criterion = nn.BCEWithLogitsLoss()

                            domain_labels_gen = domain_labels_gen.squeeze()
                            domain_preds_gen = torch.trunc(2*F.sigmoid(domain_outputs_gen).data)

                            # ---------- Pytorch 0.2.0 edit change --------------------------
                            domain_counts_gen += len(domain_preds_gen)
                            domain_epoch_corrects_gen += torch.sum(domain_preds_gen == domain_labels_gen.data)

                            domain_loss_gen = domain_criterion(domain_outputs_gen, domain_labels_gen)

                            # ------ calculate pseudo predicts and losses with weights and threshold lambda -------

                            domain_epoch_loss_gen += domain_loss_gen.data[0]

                            # ------- domain classifier update ----------

                            dom_loss_gen = 0.1*domain_loss_gen

                            for x in optimizer_list:
                                x.zero_grad()

                            model3.zero_grad()
                            dom_loss_gen.backward()
                            depth_enc_optimizer.step()
                            depth_disc_optimizer.step()

                # ---------------------------------------------------------------------
                # --------------------------- start training --------------------------
                # ---------------------------------------------------------------------
                if epoch >= CLS_EP:

                    model3.train(False)  # Set model 2 to evaluate mode
                    # model2.eval()   
                    model2.train(True) 
                    model.train(True)
                    for param in model.parameters():
                        param.requires_grad = True

                    for param in model2.parameters():
                        param.requires_grad = True

                    for p in model3.parameters():
                        p.requires_grad = False

                    draw_bool_rgb2d = False
                    while source_pointer < len(source_iter):
                        source_step_count +=1
                                
                        if args.useLargeLREpoch:
                            # base on decay
                            p = epoch / total_epochs
                            if args.MaxStep != 0:
                                p = source_step_count / args.MaxStep
                                
                            p = pull_back_to_one(p)
                            l = (2. / (1. + np.exp(-10. * p))) - 1
                            step_rate = args.decay * source_step_count
                            if (epoch == 0):
                                lr_mult = 1 / (1 + np.exp(-3*(source_step_count / len(dsets['source']))))
                            else:
                                lr_mult = (1. + step_rate)**(-0.75)
                            weight_mult = args.wt * args.wp **p

                        else:
                            # base on total epoch
                            p = (epoch - CLS_EP) / (total_epochs - CLS_EP)
                            l = (2. / (1. + np.exp(-10. * p))) - 1                        
                            if (epoch == CLS_EP):
                                lr_mult = 1 / (1 + np.exp(-3*(source_step_count / len(dsets['source']))))
                            else:
                                lr_mult = (1. + 10 * p)**(-0.75)

                        optimizer, epoch_lr_mult = cls_lr_scheduler(optimizer, lr_mult)
                        dom_optimizer, epoch_lr_mult = cls_lr_scheduler(dom_optimizer, lr_mult)
                        # dom_w_optimizer, epoch_lr_mult = dom_w_lr_scheduler(dom_w_optimizer, lr_mult, weight_mult)
                        dom_feat_optimizer, epoch_lr_mult = cls_lr_scheduler(dom_feat_optimizer, lr_mult)

                        optimizer2, epoch_lr_mult = cls_lr_scheduler(optimizer2, lr_mult)
                        dom_optimizer2, epoch_lr_mult = cls_lr_scheduler(dom_optimizer2, lr_mult)
                        # dom_w_optimizer, epoch_lr_mult = dom_w_lr_scheduler(dom_w_optimizer, lr_mult, weight_mult)
                        dom_feat_optimizer2, epoch_lr_mult = cls_lr_scheduler(dom_feat_optimizer2, lr_mult)

                        # make sure to skip the last batch if the batch length is not enough(drop last)
                        batch_count += 1
                        if (batch_count * source_batchsize > len(dsets['source'])):
                            continue
                        # ----------------- get classification input --------------------------

                        source_iter, inputs, labels, source_pointer, total_source_pointer \
                                        = iterator_update(source_iter, 'source', source_batchsize, 
                                                          source_pointer, total_source_pointer, "ori")

                        source_iter2, inputs2, labels2, source_pointer2, total_source_pointer2 \
                                                    = iterator_update(source_iter2, 'source_depth', source_batchsize2, 
                                                                      source_pointer2, total_source_pointer2, "ori")

                        if pseudo_batchsize > 0:
                        
                            pseudo_iter, pseudo_inputs, _, pseudo_labels, pseudo_pointer, total_pseudo_pointer, pseudo_weights, pseudo_dom_conf \
                                            = iterator_update(pseudo_iter, 'pseudo', 
                                                               pseudo_batchsize, pseudo_pointer, 
                                                               total_pseudo_pointer, "pseu")

                        if pseudo_batchsize2 > 0:

                            pseudo_iter2, pseudo_inputs2_rgb, pseudo_inputs2_depth, pseudo_labels2, pseudo_pointer2, total_pseudo_pointer2, pseudo_weights2, pseudo_dom_conf2 \
                                            = iterator_update(pseudo_iter2, 'pseudo_depth', 
                                                               pseudo_batchsize2, pseudo_pointer2, 
                                                               total_pseudo_pointer2, "pseu")

                        # ----------------- get domain discriminator input --------------------------

                        d_source_iter, d_inputs,  _, d_source_pointer, total_d_source_pointer \
                                        = iterator_update(d_source_iter, 'source', d_source_batchsize, 
                                                          d_source_pointer, total_d_source_pointer, "ori")

                        d_source_feat_iter, d_feat_inputs,  _, d_source_feat_pointer, total_d_source_feat_pointer \
                                        = iterator_update(d_source_feat_iter, 'source', d_source_batchsize, 
                                                          d_source_feat_pointer, total_d_source_feat_pointer, "ori")

                        d_source_iter2, d_inputs2, _, d_source_pointer2, total_d_source_pointer2 \
                                        = iterator_update(d_source_iter2, 'source_depth', d_source_batchsize2, 
                                                          d_source_pointer2, total_d_source_pointer2, "ori")

                        d_source_feat_iter2, d_feat_inputs2, _, d_source_feat_pointer2, total_d_source_feat_pointer2 \
                                        = iterator_update(d_source_feat_iter2, 'source_depth', d_source_batchsize2, 
                                                          d_source_feat_pointer2, total_d_source_feat_pointer2, "ori")


                        if d_target_batchsize > 0:
                            d_target_iter, d_target_inputs, _, d_target_pointer, total_d_target_pointer \
                                            = iterator_update(d_target_iter, 'target', d_target_batchsize, 
                                                              d_target_pointer, total_d_target_pointer, "ori")

                            d_target_feat_iter, d_target_feat_inputs, _, d_target_feat_pointer, total_d_target_feat_pointer \
                                            = iterator_update(d_target_feat_iter, 'target', d_target_batchsize, 
                                                              d_target_feat_pointer, total_d_target_feat_pointer, "ori")

                        if d_target_batchsize2 > 0:

                            if not DEPTH_GT:
                                d_target_iter2, d_target_inputs2_rgb, d_target_label2, d_target_pointer2, total_d_target_pointer2 \
                                                = iterator_update(d_target_iter2, 'target_depth_gen', d_target_batchsize2, 
                                                                  d_target_pointer2, total_d_target_pointer2, "ori")

                                d_target_feat_iter2, d_target_feat_inputs2_rgb, _, d_target_feat_pointer2, total_d_target_feat_pointer2 \
                                                = iterator_update(d_target_feat_iter2, 'target_depth_gen', d_target_batchsize2, 
                                                                  d_target_feat_pointer2, total_d_target_feat_pointer2, "ori")
                            else:                    
                                d_target_iter2, d_target_inputs2_depth, _, d_target_pointer2, total_d_target_pointer2 \
                                                = iterator_update(d_target_iter2, 'target_depth_gen', d_target_batchsize2, 
                                                                  d_target_pointer2, total_d_target_pointer2, "ori")

                                d_target_feat_iter2, d_target_feat_inputs2_depth, _, d_target_feat_pointer2, total_d_target_feat_pointer2 \
                                                = iterator_update(d_target_feat_iter2, 'target_depth_gen', d_target_batchsize2, 
                                                                  d_target_feat_pointer2, total_d_target_feat_pointer2, "ori")


                        # ----------------- get domain pseudo input --------------------------
                        if d_pseudo_source_batchsize > 0:

                            d_pseudo_source_iter, d_pseudo_source_inputs, _, d_pseudo_source_pointer, total_d_pseudo_source_pointer\
                                        = iterator_update(d_pseudo_source_iter, 'd_pseudo_source', 
                                                           d_pseudo_source_batchsize, d_pseudo_source_pointer, 
                                                           total_d_pseudo_source_pointer, "ori")
                            
                            d_pseudo_source_feat_iter, d_pseudo_source_feat_inputs, _, d_pseudo_source_feat_pointer, total_d_pseudo_source_feat_pointer \
                                        = iterator_update(d_pseudo_source_feat_iter, 'd_pseudo_source', 
                                                           d_pseudo_source_batchsize, d_pseudo_source_feat_pointer, 
                                                           total_d_pseudo_source_feat_pointer, "ori")
                            
                        if d_pseudo_all_batchsize > 0:
                            # ------------------------ T1+T2 --------------------
                            d_pseudo_all_iter, d_pseudo_all_inputs, _, _, d_pseudo_all_pointer, total_d_pseudo_all_pointer, _, d_pseudo_all_dom_conf \
                                    = iterator_update(d_pseudo_all_iter, 'd_pseudo_all', 
                                                       d_pseudo_all_batchsize, d_pseudo_all_pointer, 
                                                       total_d_pseudo_all_pointer, "pseu")

                            # ------------------------ T1+T3 --------------------
                            d_pseudo_all_feat_iter, d_pseudo_all_feat_inputs, _, _, d_pseudo_all_feat_pointer, total_d_pseudo_all_feat_pointer, _, d_pseudo_all_feat_dom_conf \
                                    = iterator_update(d_pseudo_all_feat_iter, 'd_pseudo_all_feat', 
                                                       d_pseudo_all_batchsize, d_pseudo_all_feat_pointer, 
                                                       total_d_pseudo_all_feat_pointer, "pseu")



                        if d_pseudo_source_batchsize2 > 0:

                            d_pseudo_source_iter2, d_pseudo_source_inputs2, _, d_pseudo_source_pointer2, total_d_pseudo_source_pointer2\
                                        = iterator_update(d_pseudo_source_iter2, 'd_pseudo_source_depth', 
                                                           d_pseudo_source_batchsize, d_pseudo_source_pointer2, 
                                                           total_d_pseudo_source_pointer2, "ori")
                            
                            d_pseudo_source_feat_iter2, d_pseudo_source_feat_inputs2, _, d_pseudo_source_feat_pointer2, total_d_pseudo_source_feat_pointer2 \
                                        = iterator_update(d_pseudo_source_feat_iter2, 'd_pseudo_source_depth', 
                                                           d_pseudo_source_batchsize, d_pseudo_source_feat_pointer2, 
                                                           total_d_pseudo_source_feat_pointer2, "ori")
                        if d_pseudo_all_batchsize2 > 0:
                            # ------------------------ T1+T2 --------------------
                            d_pseudo_all_iter2, d_pseudo_all_inputs2_rgb, d_pseudo_all_inputs2_depth, d_pseudo_all_labels2, d_pseudo_all_pointer2, total_d_pseudo_all_pointer2, _, d_pseudo_all_dom_conf2 \
                                    = iterator_update(d_pseudo_all_iter2, 'd_pseudo_all_depth', 
                                                       d_pseudo_all_batchsize2, d_pseudo_all_pointer2, 
                                                       total_d_pseudo_all_pointer2, "pseu")

                            # ------------------------ T1+T3 --------------------
                            d_pseudo_all_feat_iter2, d_pseudo_all_feat_inputs2_rgb, d_pseudo_all_feat_inputs2_depth, _, d_pseudo_all_feat_pointer2, total_d_pseudo_all_feat_pointer2, _, d_pseudo_all_feat_dom_conf2 \
                                    = iterator_update(d_pseudo_all_feat_iter2, 'd_pseudo_all_feat_depth', 
                                                       d_pseudo_all_batchsize2, d_pseudo_all_feat_pointer2, 
                                                       total_d_pseudo_all_feat_pointer2, "pseu")

                        # --------------------- ------------------------------- -----------------------
                        # ----------------------------- fit model ------------- -----------------------
                        # --------------------- ------------------------------- -----------------------
                        if epoch - CLS_EP < pre_epochs - CLS_EP or pseudo_batchsize <= 0:
                            # ----------- classifier inputs----------
                            fuse_inputs = inputs
                            fuse_labels = labels

                            fuse_inputs2 = inputs2
                            fuse_labels2 = labels2

                            # ----------- domain inputs----------
                            domain_inputs_tensor = torch.cat((d_inputs, d_target_inputs),0)
                            if not DEPTH_GT:
                                domain_inputs_tensor2_target = d_target_inputs2_rgb
                                domain_depth_labels = d_target_label2
                            else:
                                domain_inputs_tensor2_target = d_target_inputs2_depth
                            # domain_inputs2 = torch.cat((d_inputs2, d_target_inputs2),0)

                            domain_labels = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                                             +[0.]*int(args.batch_size / 2))
                            domain_labels2 = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                                             +[0.]*int(args.batch_size / 2))

                            dom_feat_weight = torch.FloatTensor([1.]*int(args.batch_size))
                            dom_feat_weight2 = torch.FloatTensor([1.]*int(args.batch_size))

                        else:
                            # ----------- classifier inputs----------
                            fuse_inputs = torch.cat((inputs, pseudo_inputs),0)
                            fuse_labels = torch.cat((labels, pseudo_labels),0)
                            
                            # fuse_inputs2 = torch.cat((inputs2, pseudo_inputs2),0)
                            fuse_labels2 = torch.cat((labels2, pseudo_labels2),0)

                            # ----------- domain inputs----------
                            if d_target_batchsize > 0:
                                src_weight = torch.FloatTensor([1.]*int(args.batch_size/2))
                                tgt_weight = torch.FloatTensor([1.]*d_target_batchsize)

                                dom_feat_weight = torch.cat((src_weight, d_pseudo_all_dom_conf.float(), tgt_weight),0)

                                domain_inputs_tensor = torch.cat((d_inputs, d_pseudo_source_inputs, d_pseudo_all_inputs, d_target_inputs),0)

                            else:
                                src_weight = torch.FloatTensor([1.]*int(args.batch_size/2))

                                dom_feat_weight = torch.cat((src_weight, d_pseudo_all_dom_conf.float()),0)

                                domain_inputs_tensor = torch.cat((d_inputs, d_pseudo_source_inputs, d_pseudo_all_inputs),0)

                            if d_target_batchsize2 > 0:
                                if not DEPTH_GT:
                                    domain_inputs_tensor2_target = torch.cat((d_pseudo_all_inputs2_rgb, d_target_inputs2_rgb),0)
                                    domain_depth_labels = torch.cat((d_pseudo_all_labels2, d_target_label2),0)
                                else:
                                    domain_inputs_tensor2_target = torch.cat((d_pseudo_all_inputs2_depth, d_target_inputs2_depth),0)

                            else:
                                if not DEPTH_GT:
                                    domain_inputs_tensor2_target = d_pseudo_all_inputs2_rgb
                                    domain_depth_labels = d_pseudo_all_labels2
                                else:
                                    domain_inputs_tensor2_target = d_pseudo_all_inputs2_depth


                            domain_labels = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                                             +[0.]*int(args.batch_size / 2))

                            dom_feat_weight2 = torch.FloatTensor([1.]*int(args.batch_size))
                            # if d_target_batchsize2 > 0:
                            #     src_weight2 = torch.FloatTensor([1.]*int(args.batch_size/2))
                            #     tgt_weight2 = torch.FloatTensor([1.]*d_target_batchsize2)

                            #     dom_feat_weight2 = torch.cat((src_weight2, d_pseudo_all_dom_conf2.float(), tgt_weight2),0)

                            #     domain_inputs2 = torch.cat((d_inputs2, d_pseudo_source_inputs2, d_pseudo_all_inputs2, d_target_inputs2),0)

                            # else:
                            #     src_weight2 = torch.FloatTensor([1.]*int(args.batch_size/2))

                            #     dom_feat_weight2 = torch.cat((src_weight2, d_pseudo_all_dom_conf2.float()),0)

                            #     domain_inputs2 = torch.cat((d_inputs2, d_pseudo_source_inputs2, d_pseudo_all_inputs2),0)

                            domain_labels2 = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                                             +[0.]*int(args.batch_size / 2))
                        # -------------------- train model -----------------------
                        inputs, labels = Variable(fuse_inputs.cuda()), Variable(fuse_labels.cuda())
                        labels2 = Variable(fuse_labels2.cuda())

                        domain_inputs, domain_labels = Variable(domain_inputs_tensor.cuda()), \
                                                                     Variable(domain_labels.cuda())
                        
                        domain_labels2 = Variable(domain_labels2.cuda())

                        source_weight_tensor = torch.FloatTensor([1.]*source_batchsize)
                        source_weight_tensor2 = torch.FloatTensor([1.]*source_batchsize2)

                        if pseudo_batchsize <= 0:
                            class_weights_tensor = source_weight_tensor
                            class_weights_tensor2 = source_weight_tensor2
                        else:
                            pseudo_weights_tensor = torch.FloatTensor(pseudo_weights.float())
                            pseudo_weights_tensor2 = torch.FloatTensor(pseudo_weights2.float())

                            class_weights_tensor = torch.cat((source_weight_tensor, pseudo_weights_tensor),0)
                            class_weights_tensor2 = torch.cat((source_weight_tensor2, pseudo_weights_tensor2),0)
                        
                        class_weight = Variable(class_weights_tensor.cuda())
                        class_weight2 = Variable(class_weights_tensor2.cuda())

                        # ############################################

                        # generate depth images
                        if not DEPTH_GT:
                            # gen pseudo
                            if pseudo_batchsize2 > 0:
                                pseudo_inputs_rgb2d = Variable(pseudo_inputs2_rgb.cuda(),volatile = True)
                                if USE_Conditional:
                                    pseu_np_labels = pseudo_labels2.numpy()
                                    pseu_onehot_np = np.zeros((pseu_np_labels.size, args.num_class))
                                    pseu_onehot_np[np.arange(pseu_np_labels.size), pseu_np_labels] = 1
                                    pseu_classonehot_var = Variable(torch.FloatTensor(pseu_onehot_np)).cuda()

                                    rgb2d_outputs_gen_pseudo = model3(pseudo_inputs_rgb2d, train_conditional_gen=True, classonehot=pseu_classonehot_var)

                                else:
                                    rgb2d_outputs_gen_pseudo = model3(pseudo_inputs_rgb2d, train_gen=True)

                                rgb2d_outputs_gen_pseudo = rgb2d_outputs_gen_pseudo.data.cpu()

                            # gen domain
                            if USE_Conditional:

                                t_np_labels = domain_depth_labels.numpy()
                                t_onehot_np = np.zeros((t_np_labels.size, args.num_class))
                                t_onehot_np[np.arange(t_np_labels.size), t_np_labels] = 1
                                t_classonehot_var = Variable(torch.FloatTensor(t_onehot_np)).cuda()

                                domain_inputs_rgb2d_target = Variable(domain_inputs_tensor2_target.cuda(),volatile = True)
                                rgb2d_outputs_gen_target = model3(domain_inputs_rgb2d_target, train_conditional_gen=True, classonehot=t_classonehot_var)
                            else:

                                domain_inputs_rgb2d_target = Variable(domain_inputs_tensor2_target.cuda(),volatile = True)
                                rgb2d_outputs_gen_target = model3(domain_inputs_rgb2d_target, train_gen=True)
                        

                            # visualize depth generation

                            rgb2d_outputs_gen_target = rgb2d_outputs_gen_target.data.cpu()
                            domain_inputs_rgb2d_target = domain_inputs_rgb2d_target.data.cpu()

                            if draw_bool_rgb2d == False and epoch % 10 == 0:

                                draw_rgb2d = torch.FloatTensor(domain_inputs_rgb2d_target.size(0)*2, 
                                                                 3, 
                                                                 domain_inputs_rgb2d_target.size(2), 
                                                                 domain_inputs_rgb2d_target.size(3)).fill_(0)

                                for idx in range(domain_inputs_rgb2d_target.size(0)):
                                    single_img = domain_inputs_rgb2d_target[idx,:,:,:].unsqueeze(0)


                                    recon_real = rgb2d_outputs_gen_target[idx,:,:,:].unsqueeze(0)

                                    val_inputv = single_img

                                    draw_rgb2d[idx*2+0,:,:,:].copy_(val_inputv.squeeze(0))
                                    draw_rgb2d[idx*2+1,:,:,:].copy_(recon_real.squeeze(0))
                                    # val_batch_output[idx*5+3,:,:,:].copy_(x_hat_val.data.squeeze(0))
                                    # val_batch_output[idx*5+4,:,:,:].copy_(recon_fake.data.squeeze(0))

                                vutils.save_image(draw_rgb2d, './gen/%s2%s/RGB2D_generated_epoch_%4d_iter_%05d.png' % \
                                   (args.source_set, args.target_set, epoch, source_step_count), nrow=10, normalize=True)
                                
                                draw_bool_rgb2d = True

                        else:
                            rgb2d_outputs_gen_target = domain_inputs_tensor2_target
                            if pseudo_batchsize2 > 0:
                                rgb2d_outputs_gen_pseudo = pseudo_inputs2_depth
                        # --------------------- ------------------------------- --------
                        # ------------ training classification losses ------------------
                        # --------------------- ------------------------------- --------
                        if pseudo_batchsize2 > 0:
                            inputs2 =  Variable(torch.cat((inputs2, rgb2d_outputs_gen_pseudo),0)).cuda()
                            domain_inputs_tensor2 = torch.cat((d_inputs2, d_pseudo_source_inputs2, rgb2d_outputs_gen_target),0)
                        else:
                            inputs2 = Variable(inputs2).cuda()
                            domain_inputs_tensor2 = torch.cat((d_inputs2, rgb2d_outputs_gen_target),0)
                        # if d_target_batchsize > 0:
                        #     domain_inputs_tensor2 = torch.cat((d_inputs2, d_pseudo_source_inputs2, rgb2d_outputs_gen_target),0)
                        # else:

                        domain_inputs2 = Variable(domain_inputs_tensor2).cuda()

                        # --------------------- ------------------------------- --------
                        # ------------ training classification losses ------------------
                        # --------------------- ------------------------------- --------
                        
                        # --------------------------- classification part forward ------------------------
                        class_outputs = model('cls_train', x1=inputs)
                        
                        criterion = nn.CrossEntropyLoss()

                        _, preds = torch.max(class_outputs.data, 1)
                        class_count += len(preds)
                        class_loss = compute_new_loss(class_outputs, labels, class_weight)

                        epoch_loss += class_loss.data[0]
                        total_epoch_loss += class_loss.data[0]
                        epoch_corrects += torch.sum(preds == labels.data)

                        for x in optimizer_list:
                            x.zero_grad()
                        # optimizer.zero_grad()
                        class_loss.backward()
                        optimizer.step()


                        # --------------------- ------------------------------- --------
                        # ----------- calculate domain labels and losses ---------------
                        # --------------------- ------------------------------- --------
                        
                        # ------------------------------- domain part forward ------------------------
                        domain_outputs = model('dom_train', x1=domain_inputs, l=l)

                        # , w_main, w_l1, w_l2, w_l3, l1_rev, l2_rev, l3_rev\
                                                # , 
                                                #          init_w_main=ini_w_main, init_w_l1=ini_w_l1, 
                                                #          init_w_l2=ini_w_l2, init_w_l3=ini_w_l3)

                        domain_outputs_feat = domain_outputs
                                        # , w_main_feat, w_l1_feat, w_l2_feat, w_l3_feat, _, _, _ \, 
                                        #              init_w_main=ini_w_main, init_w_l1=ini_w_l1, 
                                        #              init_w_l2=ini_w_l2, init_w_l3=ini_w_l3)

                        
                        domain_criterion = nn.BCEWithLogitsLoss()

                        domain_labels = domain_labels.squeeze()
                        domain_preds = torch.trunc(2*F.sigmoid(domain_outputs).data)

                        # domain_preds_l1 = torch.trunc(2*F.sigmoid(domain_outputs_l1).data)
                        # domain_preds_l2 = torch.trunc(2*F.sigmoid(domain_outputs_l2).data)
                        # domain_preds_l3 = torch.trunc(2*F.sigmoid(domain_outputs_l3).data)
                        correct_domain = domain_labels.data

                        # ---------- Pytorch 0.2.0 edit change --------------------------
                        domain_counts += len(domain_preds)
                        domain_epoch_corrects += torch.sum(domain_preds == correct_domain)
                        # domain_epoch_corrects_l1 += torch.sum(domain_preds_l1 == correct_domain)
                        # domain_epoch_corrects_l2 += torch.sum(domain_preds_l2 == correct_domain)
                        # domain_epoch_corrects_l3 += torch.sum(domain_preds_l3 == correct_domain)

                        domain_loss = domain_criterion(domain_outputs, domain_labels)
                        # domain_loss_l1 = domain_criterion(domain_outputs_l1, domain_labels)
                        # domain_loss_l2 = domain_criterion(domain_outputs_l2, domain_labels)
                        # domain_loss_l3 = domain_criterion(domain_outputs_l3, domain_labels)

                        # ------ calculate pseudo predicts and losses with weights and threshold lambda -------

                        domain_epoch_loss += domain_loss.data[0]
                        # domain_epoch_loss_l1 += domain_loss_l1.data[0]
                        # domain_epoch_loss_l2 += domain_loss_l2.data[0]
                        # domain_epoch_loss_l3 += domain_loss_l3.data[0]

                        # w_main = w_main.expand_as(domain_loss)
                        # w_l1 = w_l1.expand_as(domain_loss_l1)
                        # w_l2 = w_l2.expand_as(domain_loss_l2)
                        # w_l3 = w_l3.expand_as(domain_loss_l3)

                        # ------- domain classifier update ----------

                        dom_loss = abs(args.main_w) * domain_loss
                        # torch.abs(w_main)*domain_loss+ \
                        #            torch.abs(w_l1)*domain_loss_l1 + \
                        #            torch.abs(w_l2)*domain_loss_l2+ \
                        #            torch.abs(w_l3)*domain_loss_l3

                        total_epoch_loss += dom_loss.data[0]
                        for x in optimizer_list:
                            x.zero_grad()
                        # dom_optimizer.zero_grad()
                        dom_loss.backward(retain_graph=True)
                        dom_optimizer.step()

                        # ------- domain weights update ----------
                        # if epoch >= pre_epochs:
                        #     dom_w_loss = w_main*domain_loss+ \
                        #                  w_l1*domain_loss_l1+ \
                        #                  w_l2*domain_loss_l2+ \
                        #                  w_l3*domain_loss_l3
                        #     for x in optimizer_list:
                        #         x.zero_grad()

                        #     # dom_w_optimizer.zero_grad()
                        #     dom_w_loss.backward()
                        #     dom_w_optimizer.step()

                        # --------------------- ------------------------------- --------
                        # ----------------- calculate domain feat losses ---------------
                        # --------------------- ------------------------------- --------


                        # ---------- domain feature update ----------

                        dom_feat_weight_tensor = dom_feat_weight.cuda()

                        domain_feat_criterion = nn.BCEWithLogitsLoss(weight=dom_feat_weight_tensor)

                        domain_preds_feat = torch.trunc(2*F.sigmoid(domain_outputs_feat).data)
                                            
                        # domain_preds_l1_feat = torch.trunc(2*F.sigmoid(domain_outputs_l1_feat).data)
                        # domain_preds_l2_feat = torch.trunc(2*F.sigmoid(domain_outputs_l2_feat).data)
                        # domain_preds_l3_feat = torch.trunc(2*F.sigmoid(domain_outputs_l3_feat).data)
                        
                        domain_loss_feat = domain_feat_criterion(domain_outputs_feat, domain_labels)
                        # domain_loss_l1_feat = domain_feat_criterion(domain_outputs_l1_feat, domain_labels)
                        # domain_loss_l2_feat = domain_feat_criterion(domain_outputs_l2_feat, domain_labels)
                        # domain_loss_l3_feat = domain_feat_criterion(domain_outputs_l3_feat, domain_labels)

                        domain_epoch_loss += domain_loss_feat.data[0]
                        # domain_epoch_loss_l1 += domain_loss_l1_feat.data[0]
                        # domain_epoch_loss_l2 += domain_loss_l2_feat.data[0]
                        # domain_epoch_loss_l3 += domain_loss_l3_feat.data[0]

                        # w_main_feat = w_main_feat.expand_as(domain_loss_feat)
                        # w_l1_feat = w_l1_feat.expand_as(domain_loss_l1_feat)
                        # w_l2_feat = w_l2_feat.expand_as(domain_loss_l2_feat)
                        # w_l3_feat = w_l3_feat.expand_as(domain_loss_l3_feat)

                        dom_feat_loss = abs(args.main_w) * domain_loss_feat
                        # torch.abs(w_main_feat)*domain_loss_feat+ \
                        #                 torch.abs(w_l1_feat)*domain_loss_l1_feat + \
                        #                 torch.abs(w_l2_feat)*domain_loss_l2_feat+ \
                        #                 torch.abs(w_l3_feat)*domain_loss_l3_feat

                        total_epoch_loss += dom_feat_loss.data[0]
                        for x in optimizer_list:
                            x.zero_grad()
                        # dom_feat_optimizer.zero_grad()
                        dom_feat_loss.backward()
                        dom_feat_optimizer.step()
                    
                        # --------------------- ------------------------------- --------
                        # ----------- calculate domain 2 labels and losses ---------------
                        # --------------------- ------------------------------- --------

                        model.train(False)  # Set model 2 to evaluate mode
                        model.eval()   
                        model2.train(True)
                        # --------------------------- classification part modalirt2 ------------------------
                        class_outputs2 = model2('cls_train', x1=inputs2)
                        
                        criterion = nn.CrossEntropyLoss()

                        _, preds2 = torch.max(class_outputs2.data, 1)
                        class_count2 += len(preds)
                        class_loss2 = compute_new_loss(class_outputs2, labels2, class_weight2)

                        epoch_loss2 += class_loss2.data[0]
                        total_epoch_loss2 += class_loss2.data[0]
                        epoch_corrects2 += torch.sum(preds2 == labels2.data)

                        for x in optimizer_list:
                            x.zero_grad()
                        # optimizer.zero_grad()
                        class_loss2.backward()
                        optimizer2.step()

                        # ------------------- domain part forward modality 2 ------------------------ 

                        domain_outputs2 = model2('dom_train', x1=domain_inputs2, l=l)
                                                  # , 
                                                  #        init_w_main=ini_w_main_2, init_w_l1=ini_w_l1_2, 
                                                  #        init_w_l2=ini_w_l2_2, init_w_l3=ini_w_l3_2)

                        domain_outputs_feat2 = domain_outputs2
                                        # , 
                                        #              init_w_main=ini_w_main_2, init_w_l1=ini_w_l1_2, 
                                        #              init_w_l2=ini_w_l2_2, init_w_l3=ini_w_l3_2)


                        domain_loss_2 = domain_criterion(domain_outputs2, domain_labels2)
                        # domain_loss_l2_2 = domain_criterion(domain_outputs_l2_2, domain_labels2)
                        # domain_loss_l3_2 = domain_criterion(domain_outputs_l3_2, domain_labels2)
                        # domain_loss_l4_2 = domain_criterion(domain_outputs_l4_2, domain_labels2)

                        
                        domain_preds2 = torch.trunc(2*F.sigmoid(domain_outputs2)).data

                        domain_epoch_corrects2 = torch.sum(domain_preds2 == domain_labels2.data.float())
                        domain_acc_2 = domain_epoch_corrects2 * 100 / domain_preds2.size(0)
                        # domain_meter_2.update(domain_acc_2, domain_preds2.size(0))

                        # w_main_2 = w_main_2[0].expand_as(domain_loss_2)
                        # w_l2_2 = w_l2_2[0].expand_as(domain_loss_l2_2)
                        # w_l3_2 = w_l3_2[0].expand_as(domain_loss_l3_2)
                        # w_l4_2 = w_l4_2[0].expand_as(domain_loss_l4_2)


                        # ------- domain classifier update 2 ----------
                        dom_loss2 = abs(args.main_w2) * domain_loss_2
                        # torch.abs(w_main_2)*domain_loss_2+ \
                        #            torch.abs(w_l2_2)*domain_loss_l2_2 + \
                        #            torch.abs(w_l3_2)*domain_loss_l3_2 + \
                        #            torch.abs(w_l4_2)*domain_loss_l4_2
                        
                        dom_optimizer2.zero_grad()
                        dom_loss2.backward(retain_graph=True)

                        dom_optimizer2.step()

                       
                        # ------- domain weights update 2 ----------
                        # if epoch >= pre_epochs:
                        #     dom_w_loss2 = w_main_2*domain_loss_2+ \
                        #                  w_l2_2*domain_loss_l2_2+ \
                        #                  w_l3_2*domain_loss_l3_2+ \
                        #                  w_l4_2*domain_loss_l4_2
                                
                        #     dom_w_optimizer2.zero_grad()
                        #     dom_w_loss2.backward()
                        #     dom_w_optimizer2.step()

                        # --------------------- ------------------------------- --------
                        # ----------------- calculate domain 2 feat losses ---------------
                        # --------------------- ------------------------------- --------

                        # dom_feat_weight_tensor2 = dom_feat_weight2.cuda()

                        domain_feat_criterion = nn.BCEWithLogitsLoss() #weight=dom_feat_weight_tensor2

                        domain_loss_feat2 = domain_feat_criterion(domain_outputs_feat2, domain_labels2)
                        # domain_loss_l2_feat2 = domain_feat_criterion(domain_outputs_l2_feat2, domain_labels2)
                        # domain_loss_l3_feat2 = domain_feat_criterion(domain_outputs_l3_feat2, domain_labels2)
                        # domain_loss_l4_feat2 = domain_feat_criterion(domain_outputs_l4_feat2, domain_labels2)

                        # w_main_feat2 = w_main_feat2[0].expand_as(domain_loss_feat2)
                        # w_l2_feat2 = w_l2_feat2[0].expand_as(domain_loss_l2_feat2)
                        # w_l3_feat2 = w_l3_feat2[0].expand_as(domain_loss_l3_feat2)
                        # w_l4_feat2 = w_l4_feat2[0].expand_as(domain_loss_l4_feat2)


                        dom_feat_loss2 = abs(args.main_w2) * domain_loss_feat2
                        # torch.abs(w_main_feat2)*domain_loss_feat2+ \
                        #                 torch.abs(w_l2_feat2)*domain_loss_l2_feat2+ \
                        #                 torch.abs(w_l3_feat2)*domain_loss_l3_feat2+ \
                        #                 torch.abs(w_l4_feat2)*domain_loss_l4_feat2

                        for x in optimizer_list:
                            x.zero_grad()

                        dom_feat_loss2.backward()
                        dom_feat_optimizer2.step()

            # ----------------------------------------------------------
            # ------------------ Testing Phase -------------------------
            # ----------------------------------------------------------

            elif phase == 'test' :
                model.train(False)  # Set model 1 to evaluate mode
                model.eval()
                model2.train(False)  # Set model 2 to evaluate mode
                model2.eval()
                model3.train(False)  # Set model 3 to evaluate mode
                model3.eval()
                print()

                try:
                    os.makedirs('./gen/%s2%s/'%(args.source_set, args.target_set))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                for p in model3.parameters():
                    p.requires_grad = False

                draw_bool = False
                incorrect_count = 0
                # with torch.no_grad():
                for inputs, inputs2, labels, path, path2, t_dict in dset_loaders['depth_gen_draw']:
                   
                    inputs1, labels1 = Variable(inputs.cuda(),volatile=True), Variable(labels.cuda(),volatile=True)
                    labels2 = Variable(labels.cuda(),volatile=True)

                    class_outputs = model('test', inputs1)

                    _, preds = torch.max(class_outputs.data, 1)

                    if not DEPTH_GT:
                        if USE_Conditional:
                            
                            t_np_labels = preds.cpu().numpy()
                            t_onehot_np = np.zeros((t_np_labels.size, args.num_class))
                            t_onehot_np[np.arange(t_np_labels.size), t_np_labels] = 1
                            t_classonehot_var = Variable(torch.FloatTensor(t_onehot_np),volatile=True).cuda()

                            pred_depth = model3(inputs1, train_conditional_gen=True, classonehot=t_classonehot_var)
                        else:
                            pred_depth = model3(inputs1, train_gen=True)

                        class_outputs2 = model2('test', pred_depth)
                    else:
                        inputs2 = Variable(inputs2.cuda(),volatile=True)
                        class_outputs2 = model2('test', inputs2)

                    class_outputs_co = (RGB_Ratio * class_outputs.cpu() + Depth_Ratio * class_outputs2.cpu()) * (1.0 / (RGB_Ratio + Depth_Ratio))
                    # zero the parameter gradients
                    # optimizer.zero_grad()

                    # ------------ test classification statistics ------------
                    _, preds2 = torch.max(class_outputs2.data, 1)
                    _, preds_co = torch.max(class_outputs_co.data, 1)


                    criterion = nn.CrossEntropyLoss()
                    class_loss = criterion(class_outputs, labels1)
                    epoch_loss += class_loss.data[0]

                    epoch_corrects += torch.sum(preds.cpu() == labels1.cpu().data)
                    epoch_corrects2 += torch.sum(preds2.cpu() == labels2.cpu().data)
                    epoch_corrects_co += torch.sum(preds_co.cpu() == labels1.cpu().data)

                    for i in range(args.num_class):
                        if labels1.data[0] == i:
                            if i in test_corrects:
                                test_corrects[i] += torch.sum(preds.cpu() == labels1.cpu().data)
                            else:
                                test_corrects[i] = torch.sum(preds.cpu() == labels1.cpu().data)
                            if i in test_totals:
                                test_totals[i] += 1
                            else:
                                test_totals[i] = 1
                    

                    if not DEPTH_GT:
                        draw_eval = torch.FloatTensor(inputs2.size(0)*3, 
                                                             3, 
                                                             inputs2.size(2), 
                                                             inputs2.size(3)).fill_(0)

                        inputs_var = Variable(inputs.cuda(),volatile=True)

                        depth_var = Variable(inputs2.cuda(),volatile=True)

                        criterionCAE = nn.MSELoss()
                        genloss_test = criterionCAE(pred_depth, depth_var)
                        
                        loss_meter.update(genloss_test.data[0], pred_depth.size(0))
                        
                        if draw_bool == False and epoch < CLS_EP:
                            for idx in range(inputs.size(0)):
                                single_img = inputs[idx,:,:,:].unsqueeze(0)
                                target_img = inputs2[idx,:,:,:].unsqueeze(0)


                                recon_real = pred_depth[idx,:,:,:].unsqueeze(0)

                                val_inputv = single_img
                                val_targetv= target_img
                                # print(val_inputv)
                                # print(val_targetv)
                                # print(recon_real)
                                # x_hat_val  = netG(val_inputv)
                                # recon_fake = netD(x_hat_val)

                                draw_eval[idx*3+0,:,:,:].copy_(val_inputv.squeeze(0))
                                draw_eval[idx*3+1,:,:,:].copy_(val_targetv.squeeze(0))
                                draw_eval[idx*3+2,:,:,:].copy_(recon_real.data.squeeze(0))
                                # val_batch_output[idx*5+3,:,:,:].copy_(x_hat_val.data.squeeze(0))
                                # val_batch_output[idx*5+4,:,:,:].copy_(recon_fake.data.squeeze(0))

                            vutils.save_image(draw_eval, './gen/%s2%s/Test_generated_epoch_%4d_iter_%05d.png' % \
                               (args.source_set, args.target_set, epoch, source_step_count), nrow=10, normalize=True)
                            
                            draw_bool = True

                        if epoch == CLS_EP -1 :

                            fn = path[0].split('.')[0].split('/')[-1]

                            gen_class = fn.split('_')[0]

                            gen_out = torch.FloatTensor(inputs.size(0)*3, 3, inputs.size(2), inputs.size(3)).fill_(0) 

                            for idx in range(inputs.size(0)):
                                gen_out[idx*3+0,:,:,:].copy_(inputs_var.data.squeeze(0))
                                gen_out[idx*3+1,:,:,:].copy_(depth_var.data.squeeze(0))
                                gen_out[idx*3+2,:,:,:].copy_(pred_depth.data.squeeze(0))

                            try:
                                os.makedirs('./gen/%s2%s/RESULT/' %(args.source_set, args.target_set)+str(epoch)+'/'+gen_class+'/')
                            except OSError as e:
                                if e.errno != errno.EEXIST:
                                    raise
                            
                            vutils.save_image(gen_out, './gen/%s2%s/RESULT/' %(args.source_set, args.target_set)+str(epoch)+'/'+gen_class+'/'+fn+'.png', normalize=True)

            # ----------------  print statistics results   --------------------

            if phase == 'train':
                if epoch >= CLS_EP:
                    epoch_loss = epoch_loss / batch_count
                    epoch_acc = epoch_corrects / class_count
                    epoch_acc2 = epoch_corrects2 / class_count
                    epoch_loss_s = epoch_loss
                    epoch_acc_s = epoch_acc
                    epoch_acc_s2 = epoch_acc2
                    if epoch - CLS_EP < pre_epochs - CLS_EP:
                        if epoch - CLS_EP == pre_epochs - 1 - CLS_EP:
                            pre_epoch_acc_s = (pre_epoch_acc_s + epoch_acc_s) / 2
                            pre_epoch_acc_s2 = (pre_epoch_acc_s2 + epoch_acc_s2) / 2
                            pre_epoch_loss_s = (pre_epoch_loss_s + epoch_loss_s) / 2
                        else:
                            pre_epoch_acc_s = epoch_acc_s
                            pre_epoch_acc_s2 = epoch_acc_s2
                            pre_epoch_loss_s = epoch_loss_s

                    else:
                        train_num = epoch - pre_epochs + 1 
                        total_epoch_acc_s += epoch_acc_s
                        total_epoch_loss_s += epoch_loss_s
                        avg_epoch_acc_s = total_epoch_acc_s / train_num
                        avg_epoch_loss_s = total_epoch_loss_s / train_num

                        total_epoch_acc_s2 += epoch_acc_s2
                        avg_epoch_acc_s2 = total_epoch_acc_s2 / train_num

                    domain_avg_loss = domain_epoch_loss / batch_count
                    domain_avg_loss_l1 = domain_epoch_loss_l1 / batch_count
                    domain_avg_loss_l2 = domain_epoch_loss_l2 / batch_count
                    domain_avg_loss_l3 = domain_epoch_loss_l3 / batch_count

                    domain_acc = domain_epoch_corrects / domain_counts
                    domain_acc_l1 = domain_epoch_corrects_l1 / domain_counts
                    domain_acc_l2 = domain_epoch_corrects_l2 / domain_counts
                    domain_acc_l3 = domain_epoch_corrects_l3 / domain_counts
                    class_loss_point.append(float("%.4f" % epoch_loss))
                    domain_loss_point.append(float("%.4f" % domain_avg_loss))
                    source_acc_point.append(float("%.4f" % epoch_acc))
                    source_acc_point2.append(float("%.4f" % epoch_acc2))
                    domain_acc_point.append(float("%.4f" % domain_acc))
                    lr_point.append(float("%.4f" % epoch_lr_mult))

                    domain_loss_point_l1.append(float("%.4f" % domain_avg_loss_l1))
                    domain_loss_point_l2.append(float("%.4f" % domain_avg_loss_l2))
                    domain_loss_point_l3.append(float("%.4f" % domain_avg_loss_l3))

                    domain_acc_point_l1.append(float("%.4f" % domain_acc_l1))
                    domain_acc_point_l2.append(float("%.4f" % domain_acc_l2))
                    domain_acc_point_l3.append(float("%.4f" % domain_acc_l3))
                    print('Phase: {} lr_mult: {:.4f} Loss: {:.4f} D_loss: {:.4f} Acc: {:.4f} Acc2: {:.4f} D_Acc: {:.4f}'.format(
                          phase, epoch_lr_mult, epoch_loss, domain_avg_loss, 
                          epoch_acc, epoch_acc2, domain_acc))
                else:
                    gen_avg_loss = gen_epoch_loss_gen/ batch_count

                    if epoch >= POS_CLS_EP:
                        domain_acc_gen = domain_epoch_corrects_gen / domain_counts_gen
                    else:
                        domain_acc_gen = 0
                    print('Epoch {} Phase: {} DepthLoss: {:.4f} D_Acc: {:.4f}'.format(epoch, phase+"RGB2D", gen_avg_loss, domain_acc_gen))


            else:
                if epoch >= CLS_EP:
                #     # epoch_loss = epoch_loss / len(dsets['test'])'TotalAvgLoss: {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'.format(loss_meter=loss_meter)
                    epoch_acc = epoch_corrects / len(dsets['test'])
                    epoch_acc2 = epoch_corrects2 / len(dsets['test'])
                    epoch_acc_co = epoch_corrects_co / len(dsets['test'])

                    epoch_acc_t = epoch_acc
                    print('Phase: {} Loss: {:.4f} Acc: {:.4f} Acc_d: {:.4f} Acc_Fu: {:.4f} '.format(
                        phase, epoch_loss, epoch_acc, epoch_acc2, epoch_acc_co))

                    print(', '.join("%d:%d/%d"%(i,j,test_totals[i]) for i,j in test_corrects.items()))

                    target_loss_point.append(float("%.4f" % epoch_loss))
                    target_acc_point.append(float("%.4f" % epoch_acc))
                    target_acc_point2.append(float("%.4f" % epoch_acc2))
                    target_acc_point_co.append(float("%.4f" % epoch_acc_co))
                else:
                    print('Epoch {} Phase: {} Loss: {loss_meter.avg:.4f}'.format(epoch, phase+"RGB2D", loss_meter=loss_meter) )

                if epoch == total_epochs:
                    f = open("Result_Acc:"+str(epoch_acc)+"_Acc_Fu:"+str(epoch_acc_co)+".txt","w+")
                    f.close()

                print("#########################################")
            # deep copy the model, print best accuracy
            # if phase == 'test':
            #     savemod = str(args.source_set + ' 2 ' + args.target_set)
            #     # if epoch == num_epochs or epoch == int(num_epochs*1.2) or epoch == int(num_epochs* 1.5):
            #     #     save_checkpoint({
            #     #         'epoch': epoch + 1,
            #     #         'state_dict': model.state_dict(),
            #     #     }, mod=savemod, epoch=epoch)
            #     if epoch_acc > best_acc:
            #         best_acc = epoch_acc
            #     print('Best Test Accuracy: {:.4f}'.format(best_acc))

        print()

        # ------------------------------------ draw graph ------------------------------------
        # ------------------------------------------------------------------------------------

        try:
            os.makedirs('./graph/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        # ----------------------------------------------------------------

        # Create plots 2
        if epoch >= CLS_EP:
            fig, ax = plt.subplots()
            ax.plot(epoch_point, source_acc_point, 'k', label='Source Classification Accuracy',color='r')
            ax.plot(epoch_point, source_acc_point2, 'k', label='Source2 Classification Accuracy',color='m')
            ax.plot(epoch_point, domain_acc_point, 'k', label='Domain Accuracy',color='g')
            ax.plot(epoch_point, target_acc_point, 'k', label='Test Classification Accuracy',color='b')
            ax.plot(epoch_point, target_acc_point2, 'k', label='Test2 Classification Accuracy',color='c')
            ax.plot(epoch_point, target_acc_point_co, 'k', label='TestFu Classification Accuracy',color='k')

            ax.annotate("ADV " + args.source_set + ' 2 ' + args.target_set + ' 0.5 Domain', xy=(1.05, 0.7), xycoords='axes fraction')
            ax.annotate('lr: %0.4f Pre epochs: %d Max epochs: %d' % (args.base_lr, pre_epochs, total_epochs), xy=(1.05, 0.65), xycoords='axes fraction')
            # ax.annotate('Pretrain epochs: %d' % PRETRAIN_EPOCH, xy=(1.05, 0.6), xycoords='axes fraction')
            # ax.annotate('Confidence Threshold: %0.3f' % confid_threshold, xy=(1.05, 0.55), xycoords='axes fraction')
            # ax.annotSate('Discriminator Threshold: %0.3f ~ %0.3f' % (LOW_DISCRIM_THRESH_T, UP_DISCRIM_THRESH_T), xy=(1.05, 0.5), xycoords='axes fraction')
            # ax.annotate('L1,L2,L3,Main Disc_Weight: %0.4f %0.4f %0.4f %0.4f' % \
            #             (w_l1.data[0], w_l2.data[0], w_l3.data[0], -1* w_main.data[0]), xy=(1.05, 0.5), xycoords='axes fraction')

            # if epoch >= 49:
            #     ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
            # if epoch >= 99:
            #     ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
            #     ax.annotate('100 Epoch Accuracy: %0.4f' % (target_acc_point[99]), xy=(1.05, 0.3), xycoords='axes fraction')
            # if epoch >= 199:
            #     ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
            #     ax.annotate('100 Epoch Accuracy: %0.4f' % (target_acc_point[99]), xy=(1.05, 0.3), xycoords='axes fraction')
            #     ax.annotate('200 Epoch Accuracy: %0.4f' % (target_acc_point[199]), xy=(1.05, 0.25), xycoords='axes fraction')
            # if epoch >= 299:
            #     ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
            #     ax.annotate('100 Epoch Accuracy: %0.4f' % (target_acc_point[99]), xy=(1.05, 0.3), xycoords='axes fraction')
            #     ax.annotate('200 Epoch Accuracy: %0.4f' % (target_acc_point[199]), xy=(1.05, 0.25), xycoords='axes fraction')
            #     ax.annotate('300 Epoch Accuracy: %0.4f' % (target_acc_point[299]), xy=(1.05, 0.2), xycoords='axes fraction')
            # if epoch >= total_epochs:
            #     ax.annotate('%d Epoch Accuracy: %0.4f' % (int(total_epochs),target_acc_point[total_epochs-1]), xy=(1.05, 0.15), xycoords='axes fraction')

            ax.annotate('Last Epoch Accuracy: %0.4f' % (epoch_acc), xy=(1.05, 0.1), xycoords='axes fraction', size=14)

            # Now add the legend with some customizations.
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True)

            # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
            frame = legend.get_frame()
            frame.set_facecolor('0.90')

            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')

            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width

            fig.text(0.5, 0.02, 'EPOCH', ha='center')
            fig.text(0.02, 0.5, 'ACCURACY', va='center', rotation='vertical')

            plt.savefig('graph/'+args.source_set+'2'+args.target_set+'_acc.png', bbox_inches='tight')
            
            if epoch % 50 == 0 or epoch == num_epochs -1:
                try:
                    os.makedirs('./graph/'+args.source_set+'2'+args.target_set)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                plt.savefig('graph/'+args.source_set+'2'+args.target_set+'/'+'epoch'+str(epoch)+',acc'+str(epoch_acc_t)+'.png', bbox_inches='tight')

            fig.clf()

            plt.clf()

        epoch += 1
    time_elapsed = time.time() - since

    try:
        os.makedirs('./Result_txt/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return best_model, best_model2

######################################################################
# Learning rate scheduler

def save_checkpoint(state, epoch=0, mod="", filename='checkpoint.pth.tar'):
    filename = '_'.join((mod, "Epoch", str(epoch), filename))
    torch.save(state, filename)

    # if epoch!=0:
    #     filename_e = '_'.join((args.snapshot_pref, mod, "Epoch", str(epoch), 'checkpoint.pth.tar'))
    #     torch.save(state, filename_e)

def lr_scheduler(optimizer, lr_mult, weight_mult=1):
    counter = 0
    for param_group in optimizer.param_groups:
        if counter == 0:
            optimizer.param_groups[counter]['lr'] = args.base_lr * lr_mult / 10.0
        else:
            optimizer.param_groups[counter]['lr'] = args.base_lr * lr_mult
        counter += 1

    return optimizer, lr_mult

def dom_w_scheduler(optimizer, lr_mult, weight_mult=1):
    counter = 0
    for param_group in optimizer.param_groups:
        if counter == 0:
            optimizer.param_groups[counter]['lr'] = args.base_lr * lr_mult * weight_mult
        counter += 1

    return optimizer, lr_mult

def diff_states(dict_canonical, dict_subset):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    #Sanity check that param names overlap
    #Note that params are not necessarily in the same order
    #for every pretrained model
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0

    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            yield (name, v1)

def load_model_merged(name, num_classes):

    model = models.__dict__[name](num_classes=num_classes)

    #Densenets don't (yet) pass on num_classes, hack it in for 169
    if name == 'densenet169':
        model = torchvision.models.DenseNet(num_init_features=64, growth_rate=32, \
                                            block_config=(6, 12, 32, 32), num_classes=num_classes)

    if name == 'densenet201':
        model = torchvision.models.DenseNet(num_init_features=64, growth_rate=32, \
                                            block_config=(6, 12, 48, 32), num_classes=num_classes)
    if name == 'densenet161':
        model = torchvision.models.DenseNet(num_init_features=96, growth_rate=48, \
                                            block_config=(6, 12, 36, 24), num_classes=num_classes)

    pretrained_state = model_zoo.load_url(model_urls[name])

    #Diff
    diff = [s for s in diff_states(model.state_dict(), pretrained_state)]
    print("Replacing the following state from initialized", name, ":", \
          [d[0] for d in diff])

    for name, value in diff:
        pretrained_state[name] = value

    assert len([s for s in diff_states(model.state_dict(), pretrained_state)]) == 0

    #Merge
    model.load_state_dict(pretrained_state)
    return model, diff

def scale_gradients(v, weights): # assumes v is batch x ...
    def hook(g):
        return g*weights.view(*((-1,)+(len(g.size())-1)*(1,))) # probably nicer to hard-code -1,1,...,1
    v.register_hook(hook)

def compute_new_loss(logits, target, weights):
    """ logits: Unnormalized probability for each class.
        target: index of the true class(label)
        weights: weights of weighted loss.
    Returns:
        loss: An average weighted loss value
    """
    # print("l: ",logits)
    # print("t: ",target)
    weights = weights.narrow(0,0,len(target))
    # print("w: ",weights)
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size()) * weights
    # losses = losses * weights
    loss = losses.sum() / len(target)
    # length.float().sum()
    return loss

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def iterator_reset(iterator_name_i, dset_name_i, batchsize_i):
    if batchsize_i > 0:
        if args.useRatio:
            dset_loaders[iterator_name_i] = torch.utils.data.DataLoader(dsets[dset_name_i],
                                                batch_size=1, shuffle=True)
        else:
            dset_loaders[iterator_name_i] = torch.utils.data.DataLoader(dsets[dset_name_i],
                                                batch_size=batchsize_i, shuffle=True, drop_last=True)
        iterator = iter(dset_loaders[iterator_name_i])
    else:
        iterator = iter([])
    
    return iterator

def iterator_update(iterator_i, dset_name_i, batchsize_i, pointer_i, total_pointer_i,  i_type="pseu"):
    if args.useRatio:
        if pointer_i + batchsize_i > len(iterator_i):
            iterator_i = iterator_reset(iterator_i, dset_name_i, batchsize_i)
            pointer_i = 0

        iterator_batch_inputs = torch.FloatTensor([])
        iterator_batch_labels = torch.LongTensor([])
        iterator_batch_weights = torch.DoubleTensor([])
        iterator_batch_dom_confid = torch.DoubleTensor([])

        for i in range(batchsize_i):
            if i_type == "pseu":
                i_inputs, i_labels, _, i_dom_conf, i_weights, _  = iterator_i.next()
                iterator_batch_weights = torch.cat((iterator_batch_weights,i_weights), 0)
                iterator_batch_dom_confid = torch.cat((iterator_batch_dom_confid,(1-i_dom_conf)), 0)
            else:
                i_inputs, i_labels, _, _ = iterator_i.next()

            iterator_batch_inputs = torch.cat((iterator_batch_inputs,i_inputs), 0)
            iterator_batch_labels = torch.cat((iterator_batch_labels,i_labels), 0)

            pointer_i += 1
            total_pointer_i += 1
    else:
        if pointer_i +1 > len(iterator_i):
            iterator_i = iterator_reset(iterator_i, dset_name_i, batchsize_i)
            pointer_i = 0

        if i_type == "pseu":
            iterator_batch_inputs, iterator_batch_inputs_depth, iterator_batch_labels, _, i_dom_conf, iterator_batch_weights, _  = iterator_i.next()
            iterator_batch_dom_confid = 2*(1 - i_dom_conf)
        elif i_type == "depth_gen":
            iterator_batch_inputs, iterator_batch_depth_inputs, iterator_batch_labels, _, _, _ = iterator_i.next()
        else:
            iterator_batch_inputs, iterator_batch_labels, _, _ = iterator_i.next()

        pointer_i += 1
        total_pointer_i += 1

    if i_type == "pseu":
        return iterator_i, iterator_batch_inputs, iterator_batch_inputs_depth, iterator_batch_labels, pointer_i, total_pointer_i, iterator_batch_weights, iterator_batch_dom_confid
    elif i_type == "depth_gen":
        return iterator_i, iterator_batch_inputs, iterator_batch_depth_inputs, iterator_batch_labels, pointer_i, total_pointer_i
    else:
        return iterator_i, iterator_batch_inputs, iterator_batch_labels, pointer_i, total_pointer_i
                   
def pull_back_to_one(x):
    if x > 1:
        x = 1
    return x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# model_names = model_urls.keys()

#------------------------ model 1 ------------------------------

model_pretrained, diff = load_model_merged('resnet18', args.num_class)
model_pretrained2, diff2 = load_model_merged('resnet18', args.num_class)
model_pretrained3, diff3 = load_model_merged('resnet50', args.num_class)

prev_w = args.form_w
last_w = args.main_w

prev_w2 = args.form_w2
last_w2 = args.main_w2

model_dann = DISCRIMINATIVE_DANN(model_pretrained, prev_w, last_w, args.num_class)
model_dann2 = DISCRIMINATIVE_DANN(model_pretrained2, prev_w2, last_w2, args.num_class)

depth_net = DepthNet(model_pretrained3)

model_dann = model_dann.cuda()
model_dann2 = model_dann2.cuda()
depth_net = depth_net.cuda()

for param in model_dann.parameters():
    param.requires_grad = True

for param in model_dann2.parameters():
    param.requires_grad = True

for param in depth_net.parameters():
    param.requires_grad = True
# Observe that all parameters are being optimized
ignored_params_list = list(map(id, model_dann.source_bottleneck.parameters()))
ignored_params_list.extend(list(map(id, model_dann.source_classifier.parameters())))
ignored_params_list.extend(list(map(id, model_dann.domain_pred.parameters())))
# ignored_params_list.extend(list(map(id, model_dann.disc_weight.parameters())))
# ignored_params_list.extend(list(map(id, model_dann.disc_activate.parameters())))
ignored_params_list.extend(list(map(id, model_dann.domain_pred_l1.parameters())))
ignored_params_list.extend(list(map(id, model_dann.domain_pred_l2.parameters())))
ignored_params_list.extend(list(map(id, model_dann.domain_pred_l3.parameters())))
ignored_params_list.extend(list(map(id, model_dann.l1_bottleneck.parameters())))
ignored_params_list.extend(list(map(id, model_dann.l2_bottleneck.parameters())))
ignored_params_list.extend(list(map(id, model_dann.l3_bottleneck.parameters())))

base_params = filter(lambda p: id(p) not in ignored_params_list,
                     model_dann.parameters())

sub_dom_params_list = list(map(id, model_dann.domain_pred_l1.parameters()))
sub_dom_params_list.extend(list(map(id, model_dann.domain_pred_l2.parameters())))
sub_dom_params_list.extend(list(map(id, model_dann.domain_pred_l3.parameters())))
sub_dom_params_list.extend(list(map(id, model_dann.l1_bottleneck.parameters())))
sub_dom_params_list.extend(list(map(id, model_dann.l2_bottleneck.parameters())))
sub_dom_params_list.extend(list(map(id, model_dann.l3_bottleneck.parameters())))

sub_dom_params = filter(lambda p: id(p) in sub_dom_params_list,
                     model_dann.parameters())
# print(list(model_dann.parameters()))

optimizer_cls = optim.SGD([
            {'params': base_params},
            {'params': model_dann.source_bottleneck.parameters(), 'lr': args.base_lr},
            {'params': model_dann.source_classifier.parameters(), 'lr': args.base_lr},
            ], lr=args.base_lr / 10.0, momentum=0.9, weight_decay=args.decay)

optimizer_dom = optim.SGD([
            {'params': sub_dom_params},
            {'params': model_dann.source_bottleneck.parameters(), 'lr': args.base_lr},
            {'params': model_dann.domain_pred.parameters(), 'lr': args.base_lr},
            ], lr=args.base_lr / 10.0, momentum=0.9, weight_decay=args.decay)

optimizer_dom_feature = optim.SGD([
            {'params': base_params},
            ], lr=args.base_lr / 10.0, momentum=0.9, weight_decay=args.decay)

# optimizer_dom_w = optim.SGD(model_dann.disc_weight.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0)

# optimizer_pseudo = optim.SGD(model_dann.disc_activate.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=3e-4)


# Observe that all parameters are being optimized
ignored_params_list2 = list(map(id, model_dann2.source_bottleneck.parameters()))
ignored_params_list2.extend(list(map(id, model_dann2.source_classifier.parameters())))
ignored_params_list2.extend(list(map(id, model_dann2.domain_pred.parameters())))
# ignored_params_list2.extend(list(map(id, model_dann2.disc_weight.parameters())))
# ignored_params_list2.extend(list(map(id, model_dann2.disc_activate.parameters())))
ignored_params_list2.extend(list(map(id, model_dann2.domain_pred_l1.parameters())))
ignored_params_list2.extend(list(map(id, model_dann2.domain_pred_l2.parameters())))
ignored_params_list2.extend(list(map(id, model_dann2.domain_pred_l3.parameters())))
ignored_params_list2.extend(list(map(id, model_dann2.l1_bottleneck.parameters())))
ignored_params_list2.extend(list(map(id, model_dann2.l2_bottleneck.parameters())))
ignored_params_list2.extend(list(map(id, model_dann2.l3_bottleneck.parameters())))

base_params2 = filter(lambda p: id(p) not in ignored_params_list2,
                     model_dann2.parameters())

sub_dom_params_list2 = list(map(id, model_dann2.domain_pred_l1.parameters()))
sub_dom_params_list2.extend(list(map(id, model_dann2.domain_pred_l2.parameters())))
sub_dom_params_list2.extend(list(map(id, model_dann2.domain_pred_l3.parameters())))
sub_dom_params_list2.extend(list(map(id, model_dann2.l1_bottleneck.parameters())))
sub_dom_params_list2.extend(list(map(id, model_dann2.l2_bottleneck.parameters())))
sub_dom_params_list2.extend(list(map(id, model_dann2.l3_bottleneck.parameters())))

sub_dom_params2 = filter(lambda p: id(p) in sub_dom_params_list2,
                     model_dann2.parameters())
# print(list(model_dann.parameters()))

optimizer_cls2 = optim.SGD([
            {'params': base_params2},
            {'params': model_dann2.source_bottleneck.parameters(), 'lr': args.base_lr},
            {'params': model_dann2.source_classifier.parameters(), 'lr': args.base_lr},
            ], lr=args.base_lr / 10.0, momentum=0.9, weight_decay=args.decay)

optimizer_dom2 = optim.SGD([
            {'params': sub_dom_params2},
            {'params': model_dann2.source_bottleneck.parameters(), 'lr': args.base_lr},
            {'params': model_dann2.domain_pred.parameters(), 'lr': args.base_lr},
            ], lr=args.base_lr / 10.0, momentum=0.9, weight_decay=args.decay)

optimizer_dom_feature2 = optim.SGD([
            {'params': base_params2},
            ], lr=args.base_lr / 10.0, momentum=0.9, weight_decay=args.decay)


ae_list = list(map(id, depth_net.input_block.parameters()))
# ae_list.extend(list(map(id, depth_net.input_pool.parameters())))
ae_list.extend(list(map(id, depth_net.down_blocks.parameters())))
ae_list.extend(list(map(id, depth_net.down_bottleneck.parameters())))
ae_list.extend(list(map(id, depth_net.bridge.parameters())))
ae_list.extend(list(map(id, depth_net.up_blocks.parameters())))
ae_list.extend(list(map(id, depth_net.up_bottleneck.parameters())))
ae_list.extend(list(map(id, depth_net.out.parameters())))

ae_params = filter(lambda p: id(p) in ae_list, depth_net.parameters())


##############################

disc_list = list(map(id, depth_net.domain_pred.parameters()))
disc_list.extend(list(map(id, depth_net.domain_pred_conditional.parameters())))
disc_list.extend(list(map(id, depth_net.down_bottleneck.parameters())))
# disc_list.extend(list(map(id, depth_net.bridge.parameters())))
# disc_list.extend(list(map(id, depth_net.discriminate.parameters())))

disc_params = filter(lambda p: id(p) in disc_list, depth_net.parameters())

############################

enc_list = list(map(id, depth_net.input_block.parameters()))
enc_list.extend(list(map(id, depth_net.down_blocks.parameters())))
enc_params = filter(lambda p: id(p) in enc_list, depth_net.parameters())
############################

# optimizerDepth = optim.Adam(depth_net.parameters(), lr = args.lrG, betas = (args.beta1, 0.999), weight_decay=0.0003)

# optimizerEnc = optim.Adam(enc_params, lr = args.lrG, betas = (args.beta1, 0.999), weight_decay=0.0003)

optimizerGen = optim.Adam(ae_params, lr = 0.001, betas = (args.beta1, 0.999), weight_decay=0.0003)

# optimizerDisc = optim.Adam(disc_params, lr = args.lrD, betas = (args.beta1, 0.999), weight_decay=0.0003)

optimizerDepth = optim.SGD(depth_net.parameters(), lr = 0.001, momentum=0.9, weight_decay=0.0003)

optimizerEnc = optim.SGD(enc_params, lr = 0.0001, momentum=0.9, weight_decay=0.0003)

# optimizerGen = optim.SGD(ae_params, lr = args.lrG/10.0, momentum=0.9, weight_decay=0.0003)

optimizerDisc = optim.SGD(disc_params, lr = 0.001, momentum=0.9, weight_decay=0.0003)

# optimizer_dom_w2 = optim.SGD(model_dann2.disc_weight.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0)

# optimizer_pseudo2 = optim.SGD(model_dann2.disc_activate.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=3e-4)


#------------------------ train models ------------------------------

train_model(model_dann, model_dann2, depth_net, optimizer_cls, optimizer_dom, optimizer_dom_feature,
                         optimizer_cls2, optimizer_dom2, optimizer_dom_feature2, optimizerGen, optimizerDisc, optimizerDepth, 
                         optimizerEnc, lr_scheduler, dom_w_scheduler, base_params, num_epochs=total_epochs)
