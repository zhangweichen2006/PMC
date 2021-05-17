from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

import torch.nn.functional as F

from torch.autograd import Variable, Function

INI_DISC_SIG_SCALE = 0.1
INI_DISC_A = 1

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

class Contrast_ReLU_activate(nn.Module):

    def __init__(self, initWeightScale, initBias):

        super(Contrast_ReLU_activate, self).__init__()

        # self.dom_func_sigma = nn.Parameter(torch.ones(1),requires_grad=False)
        # self.integral_var = Variable(torch.FloatTensor([INTEGRAL_SIGMA_VAR]).cuda())
 
        # self.sigma_scale = initWeightScale

    def forward(self, class_t, dom_res):

        top_prob, top_label = torch.topk(F.softmax(class_t.squeeze()), 1)

        confid_rate = top_prob.squeeze()

        return confid_rate

class Discriminator_Weights_Adjust(nn.Module):

    def __init__(self, form_weight, last_weight, init_w_l2=None, init_w_l3=None, init_w_l4=None, init_w_main=None):

        super(Discriminator_Weights_Adjust, self).__init__()

        # self.main_var = Variable(torch.FloatTensor([0]).cuda())
        self.l2_var = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.l3_var = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.default_const = last_weight - form_weight

        # self.k_var = Variable(torch.FloatTensor([default_const]).cuda())

        self.f_weight = form_weight
        self.l_weight = last_weight


        self.init_w_l2 = init_w_l2
        # self.init_w_l1 = None
        self.init_w_l3 = init_w_l3
        self.init_w_l4 = init_w_l4
        self.init_w_main = init_w_main

        #  main_weight, l1_weight, l2_weight, l3_weight, init_w_main, init_w_l2, init_w_l3, init_w_l4, , l2_var, l3_var
    def forward(self):

        w_main = self.init_w_main 
        # + self.main_var
        w_l2 = self.init_w_l2 + self.l2_var
        w_l3 = self.init_w_l3 + self.l3_var

        if abs(w_l2.data[0]) > abs(self.f_weight):
            w_l2 = w_l2 - np.sign(w_l2.data[0]) * (abs(w_l2.data[0]) - abs(self.f_weight))        
        if abs(w_l3.data[0]) > abs(self.f_weight):
            w_l3 = w_l3 - np.sign(w_l3.data[0]) * (abs(w_l3.data[0]) - abs(self.f_weight))  

        w_l4 = (w_main - self.default_const) - w_l2 - w_l3      
        
        if abs(w_l4.data[0]) > abs(self.l_weight):
            w_l4 = w_l4 - np.sign(w_l4.data[0]) * (abs(w_l4.data[0]) - abs(self.l_weight)) 

        # l1_rev = torch.FloatTensor([np.sign(w_l1[0])])
        # l2_rev = torch.FloatTensor([np.sign(w_l2[0])])
        # l3_rev = torch.FloatTensor([np.sign(w_l3[0])])
        # print("w_main", w_main)
        # print("w_l3", w_l3)
        return w_main, w_l2, w_l3, w_l4
        # , l1_rev, l2_rev, l3_rev

class Modality_Weights(nn.Module):

    def __init__(self):

        super(Modality_Weights, self).__init__()

        self.c_weight = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.c_bias = torch.FloatTensor([0]).cuda()
        # nn.Parameter(torch.zeros(1),requires_grad=True)
        self.d_weight = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.d_bias = torch.FloatTensor([0]).cuda()

        self.fd_weight = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.fd_bias = torch.FloatTensor([0]).cuda()

    def forward(self):

        rgb_c_w = 1 + self.c_weight
        rgb_c_b = self.c_bias

        flow_c_w = 1 - self.c_weight
        flow_c_b = self.c_bias

        if rgb_c_w.data[0] > 1.5:
            rgb_c_w = rgb_c_w - (rgb_c_w - 1.5)
        elif rgb_c_w.data[0] < 0.5:
            rgb_c_w = rgb_c_w + (0.5 - rgb_c_w)

        if flow_c_w.data[0] > 1.5:
            flow_c_w = flow_c_w - (flow_c_w - 1.5)
        elif flow_c_w.data[0] < 0.5:
            flow_c_w = flow_c_w + (0.5 - flow_c_w)

        ####### D #########
        rgb_d_w = 1 + self.d_weight
        rgb_d_b = self.d_bias

        flow_d_w = 1 - self.d_weight
        flow_d_b = self.d_bias

        if rgb_d_w.data[0] > 1.5:
            rgb_d_w = rgb_d_w - (rgb_d_w - 1.5)
        elif rgb_d_w.data[0] < 0.5:
            rgb_d_w = rgb_d_w + (0.5 - rgb_d_w)

        if flow_d_w.data[0] > 1.5:
            flow_d_w = flow_d_w - (flow_d_w - 1.5)
        elif flow_d_w.data[0] < 0.5:
            flow_d_w = flow_d_w + (0.5 - flow_d_w)

        ####### D F #########
        rgb_df_w = 1 + self.fd_weight
        rgb_df_b = self.fd_bias

        flow_df_w = 1 - self.fd_weight
        flow_df_b = self.fd_bias

        if rgb_df_w.data[0] > 1.5:
            rgb_df_w = rgb_df_w - (rgb_df_w - 1.5)
        elif rgb_df_w.data[0] < 0.5:
            rgb_df_w = rgb_df_w + (0.5 - rgb_df_w)

        if flow_df_w.data[0] > 1.5:
            flow_df_w = flow_df_w - (flow_df_w - 1.5)
        elif flow_df_w.data[0] < 0.5:
            flow_df_w = flow_df_w + (0.5 - flow_df_w)
        # dom_prob = F.sigmoid(dom_res).squeeze()
        # dom_variance = torch.abs(dom_prob - 0.5)

        # act_weight = 0.8 + w * dom_variance**4  + b
        
        # Minimise function to zero(target)

        # add source 1 result
        # max_weight = f_weight + dom_label

        # Maximise function to one(source)
        # ones_var = Variable(torch.FloatTensor([1]*len(max_weight)).cuda())

        # out_weight = torch.min(max_weight, ones_var)
        # out_weight = f_weight

        # torch.max(out_weight, init_weight).narrow(0,0,int(BATCH_SIZE/2))

        # final_weight = Variable(torch.FloatTensor([1]*int(BATCH_SIZE/2)).cuda())

        return rgb_c_w, rgb_d_w, rgb_df_w, flow_c_w, flow_d_w, flow_df_w

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True, form_weight=0.4, last_weight=0.8, init_w_l2=None, init_w_l3=None, init_w_l4=None, init_w_main=None):
        super(TSN, self).__init__()

        # self.form_weight = form_weight
        # self.last_weight = last_weight

        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.num_class = num_class
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

        self.Modality_Weights = Modality_Weights()
        
        self.domain_pred_main = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1))
        self.domain_pred_l1 = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1))
        self.domain_pred_l2 = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1))
        self.domain_pred_l3 = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1))
        self.domain_pred_l4 = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1024), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(1024, 1))

        self.process_l1 = nn.AvgPool2d(kernel_size=56)
        self.process_l2 = nn.AvgPool2d(kernel_size=28)
        self.process_l3 = nn.AvgPool2d(kernel_size=14)
        self.process_l4 = nn.AvgPool2d(kernel_size=7)

        self.l1_bottleneck = nn.Sequential(nn.Linear(64, 256))
        self.l2_bottleneck = nn.Sequential(nn.Linear(192, 256))
        self.l3_bottleneck = nn.Sequential(nn.Linear(576, 256))
        self.l4_bottleneck = nn.Sequential(nn.Linear(1056, 256))

        self.dropout = nn.Dropout(p=self.dropout)

        self.source_bottleneck = nn.Linear(self.base_model.fc.in_features, 256)

        self.source_classifier = nn.Linear(256, self.num_class)
        if init_w_l2 is not None:
            self.disc_w = Discriminator_Weights_Adjust(form_weight, last_weight, init_w_l2, init_w_l3, init_w_l4, init_w_main)

        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):
        # feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        # if self.dropout == 0:
        #     setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
        #     self.new_fc = None
        # else:
        #     setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        #     self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        # if self.new_fc is None:
        #     normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
        #     constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        # else:
        normal(self.source_classifier.weight, 0, std)
        constant(self.source_classifier.bias, 0)

        # return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            # print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        disc_weight_param = []

        CD_weight_param = []


        source_bottle_w = []
        source_bottle_b = []

        sub_bottle_w = []
        sub_bottle_b = []

        source_classifier_w = []
        source_classifier_b = []

        main_domain_classifier_weight = []
        main_domain_classifier_bias = []
        
        sub_domain_classifier_weight = []
        sub_domain_classifier_bias = []
        
        conv_cnt = 0
        bn_cnt = 0
        for m_name, m in self.named_modules():
            # if 'disc_weight' in m_name:
            #     disc_weight_param = m.parameters()

            if 'disc' in m_name:
                ps = list(m.parameters())
                disc_weight_param.append(ps[0])
                disc_weight_param.append(ps[1])

            elif 'Modality_Weights' in m_name:
                ps = list(m.parameters())
                CD_weight_param.append(ps[0])
                CD_weight_param.append(ps[1])
                CD_weight_param.append(ps[2])

            elif 'source_bottleneck' in m_name:
                ps = list(m.parameters())
                source_bottle_w.append(ps[0])
                if len(ps) == 2:
                    source_bottle_b.append(ps[1])

            elif 'bottleneck' in m_name:
                ps = list(m.parameters())
                sub_bottle_w.append(ps[0])
                if len(ps) == 2:
                    sub_bottle_b.append(ps[1])

            elif 'source_classifier' in m_name:
                ps = list(m.parameters())
                source_classifier_w.append(ps[0])
                if len(ps) == 2:
                    source_classifier_b.append(ps[1])

            elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if 'domain' in m_name:
                    if 'domain_main' in m_name:
                        main_domain_classifier_weight.append(ps[0])
                        if len(ps) == 2:
                            main_domain_classifier_bias.append(ps[1])
                    else:
                    # print("m_name",m_name)
                        sub_domain_classifier_weight.append(ps[0])
                        if len(ps) == 2:
                            sub_domain_classifier_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"}, 
            ], \
            [{'params': source_bottle_w, 'lr_mult': 1, 'decay_mult': 1,
             'name': "source_bottle_w"},
            {'params': source_bottle_b, 'lr_mult': 2, 'decay_mult': 0,
             'name': "source_bottle_b"}
            ], \
            [{'params': sub_bottle_w, 'lr_mult': 1, 'decay_mult': 1,
             'name': "sub_bottle_w"},
            {'params': sub_bottle_b, 'lr_mult': 2, 'decay_mult': 0,
             'name': "sub_bottle_b"}
            ], \
            [{'params': source_classifier_w, 'lr_mult': 1, 'decay_mult': 1,
             'name': "source_classifier_w"},
            {'params': source_classifier_b, 'lr_mult': 2, 'decay_mult': 0,
             'name': "source_classifier_b"}
            ], \
            [{'params': main_domain_classifier_weight, 'lr_mult': 10, 'decay_mult': 1,
             'name': "domain_classifier_weight"},
            {'params': main_domain_classifier_bias, 'lr_mult': 20, 'decay_mult': 0,
             'name': "domain_classifier_bias"}, 
            {'params': sub_domain_classifier_weight, 'lr_mult': 10, 'decay_mult': 1,
             'name': "domain_classifier_weight"},
            {'params': sub_domain_classifier_bias, 'lr_mult': 20, 'decay_mult': 0,
             'name': "domain_classifier_bias"}
            ], \
            [{'params': disc_weight_param, 'lr_mult': 1, 'decay_mult': 0,
             'name': "disc_weight_param"}
            ], \
            [{'params': CD_weight_param, 'lr_mult': 1, 'decay_mult': 0,
             'name': "CD_weight_param"}
            ]

    def forward(self, cond, input=None, input2=None, l=None):
     # init_w_main=None, init_w_l1=None,
     #            init_w_l2=None, init_w_l3=None):

        #######################################
        # base

        if cond == 'CD_train':

            rgb_c_w, rgb_d_w, rgb_df_w, flow_c_w, flow_d_w, flow_df_w = self.Modality_Weights() 

            return rgb_c_w, rgb_d_w, rgb_df_w, flow_c_w, flow_d_w, flow_df_w

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        base_l1 = self.layer1(input.view((-1, sample_len) + input.size()[-2:]))
        base_l2 = self.layer2(base_l1)
        base_l3 = self.layer3(base_l2)
        base_l4 = self.layer4(base_l3)
        base_out = self.layer5(base_l4)  

        # dropout layer
        if self.dropout > 0:
            base_out = self.dropout(base_out)

        #######################################
        # domain discriminator
        if (cond == 'pseudo_discriminator'): 

            bottle = self.source_bottleneck(base_out)
            class_pred = self.source_classifier(bottle)

            if not self.before_softmax:
                class_pred = self.softmax(class_pred)
            
            if self.reshape:
                class_pred = class_pred.view((-1, self.num_segments) + class_pred.size()[1:])

            class_pred = self.consensus(class_pred)


            dom_pred = self.domain_pred_main(bottle)

            if self.reshape:
                dom_pred = dom_pred.view((-1, self.num_segments) + dom_pred.size()[1:])

            dom_pred = self.consensus(dom_pred)

            return class_pred.squeeze(1), dom_pred.squeeze()

        elif (cond == 'pretrain'): 

            bottle = self.source_bottleneck(base_out)
            class_pred = self.source_classifier(bottle)

            if not self.before_softmax:
                class_pred = self.softmax(class_pred)
            
            if self.reshape:
                class_pred = class_pred.view((-1, self.num_segments) + class_pred.size()[1:])

            class_pred = self.consensus(class_pred)

            # domain classification
            # base_out_d = self.base_model(input2.view((-1, sample_len) + input2.size()[-2:]))
            
            base_l1_d = self.layer1(input2.view((-1, sample_len) + input.size()[-2:]))
            base_l2_d = self.layer2(base_l1_d)
            base_l3_d = self.layer3(base_l2_d)
            base_l4_d = self.layer4(base_l3_d)
            base_l5_d = self.layer5(base_l4_d)   

            if self.dropout > 0:
                base_l5_d = self.dropout(base_l5_d)

            bottle_l5 = self.source_bottleneck(base_l5_d)

            # dropout layer
            # process_l2 = self.process_l2(base_l2_d).view(base_l2_d.size(0), -1)
            # bottle_l2 = self.l2_bottleneck(process_l2)

            # process_l3 = self.process_l3(base_l3_d).view(base_l3_d.size(0), -1)
            # bottle_l3 = self.l3_bottleneck(process_l3)

            # process_l4 = self.process_l4(base_l4_d).view(base_l4_d.size(0), -1)
            # bottle_l4 = self.l4_bottleneck(process_l4)

            # disc_main, disc_l2, disc_l3, disc_l4 \
            #         = self.disc_w(self.init_w_main, self.init_w_l2,
            #                                        self.init_w_l3, self.init_w_l4, self.l2_var, self.l3_var)

            bottle_reverse = grad_reverse(bottle_l5, l*-1)
            # bottle_l2 = grad_reverse(bottle_l2, l*l2_rev)
            # bottle_l3 = grad_reverse(bottle_l3, l*l3_rev)
            # bottle_l4 = grad_reverse(bottle_l4, l*l4_rev)

            dom_pred = self.domain_pred_main(bottle_reverse)
            # dom_pred_l2 = self.domain_pred_l2(bottle_l2)
            # dom_pred_l3 = self.domain_pred_l3(bottle_l3)
            # dom_pred_l4 = self.domain_pred_l4(bottle_l4)

            if self.reshape:
                dom_pred = dom_pred.view((-1, self.num_segments) + dom_pred.size()[1:])
                # dom_pred_l2 = dom_pred_l2.view((-1, self.num_segments) + dom_pred_l2.size()[1:])
                # dom_pred_l3 = dom_pred_l3.view((-1, self.num_segments) + dom_pred_l3.size()[1:])
                # dom_pred_l4 = dom_pred_l4.view((-1, self.num_segments) + dom_pred_l4.size()[1:])

            dom_pred = self.consensus(dom_pred)
            # dom_pred_l2 = self.consensus(dom_pred_l2)
            # dom_pred_l3 = self.consensus(dom_pred_l3)
            # dom_pred_l4 = self.consensus(dom_pred_l4)

            return class_pred.squeeze(1), dom_pred.squeeze()
            # , dom_pred_l2.squeeze(), dom_pred_l3.squeeze(), \
            #                               dom_pred_l4.squeeze(),disc_main, disc_l2, disc_l3, disc_l4, l2_rev, l3_rev, l4_rev

        if cond == 'dom_train':

            bottle = self.source_bottleneck(base_out)

            # process_l1 = self.process_l1(base_l1).view(l1.size(0), -1)
            # bottle_l1 = self.l1_bottleneck(process_l1)

            # process_l2 = self.process_l2(base_l2).view(base_l2.size(0), -1)
            # bottle_l2 = self.l2_bottleneck(process_l2)

            # process_l3 = self.process_l3(base_l3).view(base_l3.size(0), -1)
            # bottle_l3 = self.l3_bottleneck(process_l3)

            # process_l4 = self.process_l4(base_l4).view(base_l4.size(0), -1)
            # bottle_l4 = self.l4_bottleneck(process_l4)


            # disc_main, disc_l2, disc_l3, disc_l4 \
            #         = self.disc_w()

            # print("disc_main", disc_main) self.init_w_main, self.init_w_l2, self.l2_var, self.l3_var
                                                   # self.init_w_l3, self.init_w_l4, 

            # l2_rev = np.sign(disc_l2.data[0])
            # l3_rev = np.sign(disc_l3.data[0])
            # l4_rev = np.sign(disc_l4.data[0])

            bottle = grad_reverse(bottle, l*-1)
            # bottle_l2 = grad_reverse(bottle_l2, l*l2_rev)
            # bottle_l3 = grad_reverse(bottle_l3, l*l3_rev)
            # bottle_l4 = grad_reverse(bottle_l4, l*l4_rev)

            dom_pred = self.domain_pred_main(bottle)
            # dom_pred_l2 = self.domain_pred_l2(bottle_l2)
            # dom_pred_l3 = self.domain_pred_l3(bottle_l3)
            # dom_pred_l4 = self.domain_pred_l4(bottle_l4)

            if self.reshape:
                dom_pred = dom_pred.view((-1, self.num_segments) + dom_pred.size()[1:])
                # dom_pred_l2 = dom_pred_l2.view((-1, self.num_segments) + dom_pred_l2.size()[1:])
                # dom_pred_l3 = dom_pred_l3.view((-1, self.num_segments) + dom_pred_l3.size()[1:])
                # dom_pred_l4 = dom_pred_l4.view((-1, self.num_segments) + dom_pred_l4.size()[1:])

            dom_pred = self.consensus(dom_pred)
            # dom_pred_l2 = self.consensus(dom_pred_l2)
            # dom_pred_l3 = self.consensus(dom_pred_l3)
            # dom_pred_l4 = self.consensus(dom_pred_l4)

            return dom_pred.squeeze()
            # , dom_pred_l2.squeeze(), \
            #        dom_pred_l3.squeeze(), dom_pred_l4.squeeze(), \
            #        disc_main, disc_l2, disc_l3, disc_l4
                   # , l2_rev, l3_rev, l4_rev

        #######################################
        # classification only
        else:

            bottle = self.source_bottleneck(base_out)
            class_pred = self.source_classifier(bottle)

            if not self.before_softmax:
                class_pred = self.softmax(class_pred)
            
            if self.reshape:
                class_pred = class_pred.view((-1, self.num_segments) + class_pred.size()[1:])

            class_pred = self.consensus(class_pred)

            return class_pred.squeeze(1)


    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

    def layer1(self, input):
        conv1_7x7_s2_out = self.base_model.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.base_model.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.base_model.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.base_model.pool1_3x3_s2(conv1_relu_7x7_out)

        return pool1_3x3_s2_out

    def layer2(self, input):
        conv2_3x3_reduce_out = self.base_model.conv2_3x3_reduce(input)
        conv2_3x3_reduce_bn_out = self.base_model.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.base_model.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.base_model.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.base_model.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.base_model.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.base_model.pool2_3x3_s2(conv2_relu_3x3_out)

        return pool2_3x3_s2_out

    def layer3(self, input):
        inception_3a_1x1_out = self.base_model.inception_3a_1x1(input)
        inception_3a_1x1_bn_out = self.base_model.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.base_model.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.base_model.inception_3a_3x3_reduce(input)
        inception_3a_3x3_reduce_bn_out = self.base_model.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.base_model.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.base_model.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.base_model.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.base_model.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.base_model.inception_3a_double_3x3_reduce(input)
        inception_3a_double_3x3_reduce_bn_out = self.base_model.inception_3a_double_3x3_reduce_bn(inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.base_model.inception_3a_relu_double_3x3_reduce(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.base_model.inception_3a_double_3x3_1(inception_3a_relu_double_3x3_reduce_out)
        inception_3a_double_3x3_1_bn_out = self.base_model.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.base_model.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.base_model.inception_3a_double_3x3_2(inception_3a_relu_double_3x3_1_out)
        inception_3a_double_3x3_2_bn_out = self.base_model.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.base_model.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.base_model.inception_3a_pool(input)
        inception_3a_pool_proj_out = self.base_model.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.base_model.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.base_model.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_relu_1x1_out,inception_3a_relu_3x3_out,inception_3a_relu_double_3x3_2_out ,inception_3a_relu_pool_proj_out], 1)
        inception_3b_1x1_out = self.base_model.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.base_model.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.base_model.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.base_model.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.base_model.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.base_model.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.base_model.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.base_model.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.base_model.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.base_model.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.base_model.inception_3b_double_3x3_reduce_bn(inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.base_model.inception_3b_relu_double_3x3_reduce(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.base_model.inception_3b_double_3x3_1(inception_3b_relu_double_3x3_reduce_out)
        inception_3b_double_3x3_1_bn_out = self.base_model.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.base_model.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.base_model.inception_3b_double_3x3_2(inception_3b_relu_double_3x3_1_out)
        inception_3b_double_3x3_2_bn_out = self.base_model.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.base_model.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.base_model.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.base_model.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.base_model.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.base_model.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_relu_1x1_out,inception_3b_relu_3x3_out,inception_3b_relu_double_3x3_2_out,inception_3b_relu_pool_proj_out], 1)
        inception_3c_3x3_reduce_out = self.base_model.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.base_model.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.base_model.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.base_model.inception_3c_3x3(inception_3c_relu_3x3_reduce_out)
        inception_3c_3x3_bn_out = self.base_model.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.base_model.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.base_model.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.base_model.inception_3c_double_3x3_reduce_bn(inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.base_model.inception_3c_relu_double_3x3_reduce(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.base_model.inception_3c_double_3x3_1(inception_3c_relu_double_3x3_reduce_out)
        inception_3c_double_3x3_1_bn_out = self.base_model.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = self.base_model.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.base_model.inception_3c_double_3x3_2(inception_3c_relu_double_3x3_1_out)
        inception_3c_double_3x3_2_bn_out = self.base_model.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = self.base_model.inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out)
        inception_3c_pool_out = self.base_model.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_relu_3x3_out,inception_3c_relu_double_3x3_2_out,inception_3c_pool_out], 1)

        return inception_3c_output_out

    def layer4(self, input):
        inception_4a_1x1_out = self.base_model.inception_4a_1x1(input)
        inception_4a_1x1_bn_out = self.base_model.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.base_model.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.base_model.inception_4a_3x3_reduce(input)
        inception_4a_3x3_reduce_bn_out = self.base_model.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.base_model.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.base_model.inception_4a_3x3(inception_4a_relu_3x3_reduce_out)
        inception_4a_3x3_bn_out = self.base_model.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.base_model.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.base_model.inception_4a_double_3x3_reduce(input)
        inception_4a_double_3x3_reduce_bn_out = self.base_model.inception_4a_double_3x3_reduce_bn(inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.base_model.inception_4a_relu_double_3x3_reduce(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.base_model.inception_4a_double_3x3_1(inception_4a_relu_double_3x3_reduce_out)
        inception_4a_double_3x3_1_bn_out = self.base_model.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = self.base_model.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.base_model.inception_4a_double_3x3_2(inception_4a_relu_double_3x3_1_out)
        inception_4a_double_3x3_2_bn_out = self.base_model.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = self.base_model.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
        inception_4a_pool_out = self.base_model.inception_4a_pool(input)
        inception_4a_pool_proj_out = self.base_model.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.base_model.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.base_model.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_relu_1x1_out,inception_4a_relu_3x3_out,inception_4a_relu_double_3x3_2_out,inception_4a_relu_pool_proj_out], 1)
        inception_4b_1x1_out = self.base_model.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.base_model.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.base_model.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.base_model.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.base_model.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.base_model.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.base_model.inception_4b_3x3(inception_4b_relu_3x3_reduce_out)
        inception_4b_3x3_bn_out = self.base_model.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.base_model.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.base_model.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.base_model.inception_4b_double_3x3_reduce_bn(inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.base_model.inception_4b_relu_double_3x3_reduce(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.base_model.inception_4b_double_3x3_1(inception_4b_relu_double_3x3_reduce_out)
        inception_4b_double_3x3_1_bn_out = self.base_model.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = self.base_model.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.base_model.inception_4b_double_3x3_2(inception_4b_relu_double_3x3_1_out)
        inception_4b_double_3x3_2_bn_out = self.base_model.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = self.base_model.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
        inception_4b_pool_out = self.base_model.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.base_model.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.base_model.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.base_model.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_relu_1x1_out,inception_4b_relu_3x3_out,inception_4b_relu_double_3x3_2_out,inception_4b_relu_pool_proj_out], 1)
        inception_4c_1x1_out = self.base_model.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.base_model.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.base_model.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.base_model.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.base_model.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.base_model.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.base_model.inception_4c_3x3(inception_4c_relu_3x3_reduce_out)
        inception_4c_3x3_bn_out = self.base_model.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.base_model.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.base_model.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.base_model.inception_4c_double_3x3_reduce_bn(inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.base_model.inception_4c_relu_double_3x3_reduce(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.base_model.inception_4c_double_3x3_1(inception_4c_relu_double_3x3_reduce_out)
        inception_4c_double_3x3_1_bn_out = self.base_model.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = self.base_model.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.base_model.inception_4c_double_3x3_2(inception_4c_relu_double_3x3_1_out)
        inception_4c_double_3x3_2_bn_out = self.base_model.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = self.base_model.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out)
        inception_4c_pool_out = self.base_model.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.base_model.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.base_model.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.base_model.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_relu_1x1_out,inception_4c_relu_3x3_out,inception_4c_relu_double_3x3_2_out,inception_4c_relu_pool_proj_out], 1)
        inception_4d_1x1_out = self.base_model.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.base_model.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.base_model.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.base_model.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.base_model.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.base_model.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.base_model.inception_4d_3x3(inception_4d_relu_3x3_reduce_out)
        inception_4d_3x3_bn_out = self.base_model.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.base_model.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.base_model.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.base_model.inception_4d_double_3x3_reduce_bn(inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.base_model.inception_4d_relu_double_3x3_reduce(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.base_model.inception_4d_double_3x3_1(inception_4d_relu_double_3x3_reduce_out)
        inception_4d_double_3x3_1_bn_out = self.base_model.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = self.base_model.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.base_model.inception_4d_double_3x3_2(inception_4d_relu_double_3x3_1_out)
        inception_4d_double_3x3_2_bn_out = self.base_model.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = self.base_model.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
        inception_4d_pool_out = self.base_model.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.base_model.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.base_model.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.base_model.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_relu_1x1_out,inception_4d_relu_3x3_out,inception_4d_relu_double_3x3_2_out,inception_4d_relu_pool_proj_out], 1)
        inception_4e_3x3_reduce_out = self.base_model.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.base_model.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.base_model.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.base_model.inception_4e_3x3(inception_4e_relu_3x3_reduce_out)
        inception_4e_3x3_bn_out = self.base_model.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.base_model.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.base_model.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.base_model.inception_4e_double_3x3_reduce_bn(inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.base_model.inception_4e_relu_double_3x3_reduce(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.base_model.inception_4e_double_3x3_1(inception_4e_relu_double_3x3_reduce_out)
        inception_4e_double_3x3_1_bn_out = self.base_model.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = self.base_model.inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.base_model.inception_4e_double_3x3_2(inception_4e_relu_double_3x3_1_out)
        inception_4e_double_3x3_2_bn_out = self.base_model.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = self.base_model.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
        inception_4e_pool_out = self.base_model.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_relu_3x3_out,inception_4e_relu_double_3x3_2_out,inception_4e_pool_out], 1)

        return inception_4e_output_out

    def layer5(self, input):
        inception_5a_1x1_out = self.base_model.inception_5a_1x1(input)
        inception_5a_1x1_bn_out = self.base_model.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.base_model.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.base_model.inception_5a_3x3_reduce(input)
        inception_5a_3x3_reduce_bn_out = self.base_model.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.base_model.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.base_model.inception_5a_3x3(inception_5a_relu_3x3_reduce_out)
        inception_5a_3x3_bn_out = self.base_model.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.base_model.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.base_model.inception_5a_double_3x3_reduce(input)
        inception_5a_double_3x3_reduce_bn_out = self.base_model.inception_5a_double_3x3_reduce_bn(inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.base_model.inception_5a_relu_double_3x3_reduce(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.base_model.inception_5a_double_3x3_1(inception_5a_relu_double_3x3_reduce_out)
        inception_5a_double_3x3_1_bn_out = self.base_model.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = self.base_model.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.base_model.inception_5a_double_3x3_2(inception_5a_relu_double_3x3_1_out)
        inception_5a_double_3x3_2_bn_out = self.base_model.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = self.base_model.inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out)
        inception_5a_pool_out = self.base_model.inception_5a_pool(input)
        inception_5a_pool_proj_out = self.base_model.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.base_model.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.base_model.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_relu_1x1_out,inception_5a_relu_3x3_out,inception_5a_relu_double_3x3_2_out,inception_5a_relu_pool_proj_out], 1)
        inception_5b_1x1_out = self.base_model.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.base_model.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.base_model.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.base_model.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.base_model.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.base_model.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.base_model.inception_5b_3x3(inception_5b_relu_3x3_reduce_out)
        inception_5b_3x3_bn_out = self.base_model.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.base_model.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.base_model.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.base_model.inception_5b_double_3x3_reduce_bn(inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.base_model.inception_5b_relu_double_3x3_reduce(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.base_model.inception_5b_double_3x3_1(inception_5b_relu_double_3x3_reduce_out)
        inception_5b_double_3x3_1_bn_out = self.base_model.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = self.base_model.inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.base_model.inception_5b_double_3x3_2(inception_5b_relu_double_3x3_1_out)
        inception_5b_double_3x3_2_bn_out = self.base_model.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = self.base_model.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
        inception_5b_pool_out = self.base_model.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.base_model.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.base_model.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.base_model.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_relu_1x1_out,inception_5b_relu_3x3_out,inception_5b_relu_double_3x3_2_out,inception_5b_relu_pool_proj_out], 1)
        inception_5b_output_out = self.base_model.global_pool(inception_5b_output_out)
        inception_5b_output_out = inception_5b_output_out.view(inception_5b_output_out.size(0), -1)

        return inception_5b_output_out
