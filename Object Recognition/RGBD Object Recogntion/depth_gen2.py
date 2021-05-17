import torchvision
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable, Function

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() #Leaky
        self.with_nonlinearity = with_nonlinearity
        self.use_bn = use_bn
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn: 
            x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)



class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super(UpBlockForUNetWithResNet50, self).__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

# class DepthDiscriminatorBlock(nn.Module):
#     """
#     Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
#     """

#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
#         super(DepthDiscriminatorBlock, self).__init__()

#         self.discrim_conv = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
#         # self.discrim_relu = nn.LeakyReLU()
#         # self.discrim_drop = nn.Dropout()

#     def forward(self, x):
#         """
#         :param up_x: this is the output from the previous up block
#         :param down_x: this is the output from the down block
#         :return: upsampled feature map
#         """
#         x = self.discrim_conv(x)
#         # x = self.discrim_relu(x)
#         # x = self.discrim_drop(x)
#         return x



# net = lrelu(slim.conv2d(images, 64, 5, stride=2, normalizer_fn=None))
#                 net = lrelu(slim.conv2d(net, 128, 5, stride=2))
#                 net = lrelu(slim.conv2d(net, 256, 5, stride=2))
#                 net = lrelu(slim.conv2d(net, 512, 5, stride=2)) # shape = (batch, 7, 7, 512)
# net = tf.reshape(net, [-1, (112 / 2**4)**2 * 512])

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class DepthNet(nn.Module):

    def __init__(self, pre_trained, n_channels=3, n_classes=10, depth = 6, ksize=7):
        super(DepthNet, self).__init__()
        # resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        discriminator_blocks = []
        self.ksize = ksize
        self.relu = nn.ReLU()
        self.depth = depth
        self.input_block = nn.Sequential(*list(pre_trained.children())[:3])
        self.input_pool = list(pre_trained.children())[3]
        for bottleneck in list(pre_trained.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.avgpool = pre_trained.avgpool

        # self.down = nn.Linear(2048*7*7, )
        self.down_bottleneck = nn.Linear(pre_trained.fc.in_features, 256)


        self.up_bottleneck = nn.Linear(256 + n_classes, pre_trained.fc.in_features * ksize * ksize)

        # self.up_dec = nn.Linear(pre_trained.fc.in_features, pre_trained.fc.in_features * 7 * 7)

        self.bridge = Bridge(2048, 2048)


        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_channels, kernel_size=1, stride=1)

        # discriminator_blocks.append(DepthDiscriminatorBlock(3, 64))
        # discriminator_blocks.append(DepthDiscriminatorBlock(64, 128))
        # discriminator_blocks.append(DepthDiscriminatorBlock(128, 256))
        # discriminator_blocks.append(DepthDiscriminatorBlock(256, 512))

        # self.discriminator_blocks

        # self.discriminator_blocks = nn.Sequential(ConvBlock(3, 64, padding=1, kernel_size=5, stride=2, use_bn=False),
        #                                           ConvBlock(64, 128, padding=1, kernel_size=5, stride=2),
        #                                           ConvBlock(128, 256, padding=1, kernel_size=5, stride=2),
        #                                           ConvBlock(256, 512, padding=1, kernel_size=5, stride=2),
        #                                           ConvBlock(512, 512, padding=1, kernel_size=5, stride=2))

        # self.discriminate = nn.Linear(6*6*512, 1)

        self.domain_pred = nn.Sequential(nn.Linear(256, 2048), nn.ReLU(True), nn.Dropout(),
                                         # nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))

        # # self.discriminator = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(0.2),
        #                                  nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(0.2),
        #                                  nn.Linear(2048, 1))

    def forward(self, x, x2=None, train_discrim=False, train_gen=False, train_conditional_gen=False, train_net=False, train_dom=False, train_conditional_dom=False, with_output_feature_map=False, l=1, classonehot=[]):

        # if train_discrim:

        #     dis = self.discriminator_blocks(x).squeeze()
        #     # dis = dis.view(-1, 6*6*512)
        #     # dis = self.discriminate(dis)

        #     return dis

        if train_gen:
            pre_pools = dict()
            pre_pools["layer_0"] = x
            x = self.input_block(x)
            pre_pools["layer_1"] = x
            x = self.input_pool(x)

            for i, block in enumerate(self.down_blocks, 2):
                x = block(x)
                if i == (self.depth - 1):
                    continue
                pre_pools["layer_"+str(i)] = x

            # z = self.avgpool(x)
            x = self.bridge(x)

            for i, block in enumerate(self.up_blocks, 1):
                k = self.depth - 1 - i
                key = "layer_"+str(k)
                x = block(x, pre_pools[key])
            output_feature_map = x
            depth_gen = self.out(x)
            del pre_pools


            if with_output_feature_map:
                return depth_gen, output_feature_map
            else:
                return depth_gen

        elif train_conditional_gen:
            pre_pools = dict()
            pre_pools["layer_0"] = x
            x = self.input_block(x)
            pre_pools["layer_1"] = x
            x = self.input_pool(x)

            for i, block in enumerate(self.down_blocks, 2):
                x = block(x)
                if i == (self.depth - 1):
                    continue
                pre_pools["layer_"+str(i)] = x

            x = self.avgpool(x)
            x = x.view(x.size(0),-1)
            x = self.down_bottleneck(x)
            class_x = torch.cat([classonehot,x], dim=1)
            class_x = self.up_bottleneck(class_x)
            class_x = self.relu(class_x)
            class_x = class_x.view(class_x.size(0),-1, self.ksize, self.ksize)
            # print("Before Bridge:",x)
            # x = self.bridge(x)
            # print("After Bridge:",x)

            for i, block in enumerate(self.up_blocks, 1):
                k = self.depth - 1 - i
                key = "layer_"+str(k)
                class_x = block(class_x, pre_pools[key])
            output_feature_map = class_x
            depth_gen = self.out(class_x)
            del pre_pools


            if with_output_feature_map:
                return depth_gen, output_feature_map
            else:
                return depth_gen
        
        elif train_dom:
            pre_pools = dict()
            pre_pools["layer_0"] = x
            x = self.input_block(x)
            pre_pools["layer_1"] = x
            x = self.input_pool(x)

            for i, block in enumerate(self.down_blocks, 2):
                x = block(x)
                if i == (self.depth - 1):
                    continue
                pre_pools["layer_"+str(i)] = x

            # print("x",x)
            # bridge = self.bridge(x)
            bridge = self.avgpool(x)
            bridge = bridge.view(bridge.size(0),-1)
            # print("bridge",bridge)
            bridge_reverse = grad_reverse(bridge, l*-1)
            # print("bridge_reverse",bridge_reverse)
            dom_pred = self.domain_pred(bridge_reverse)


            # for i, block in enumerate(self.up_blocks, 1):
            #     k = self.depth - 1 - i
            #     key = "layer_"+str(k)
            #     x = block(x, pre_pools[key])
            # output_feature_map = x
            # depth_gen = self.out(x)
            # del pre_pools

        elif train_conditional_dom:
            pre_pools = dict()
            pre_pools["layer_0"] = x
            x = self.input_block(x)
            pre_pools["layer_1"] = x
            x = self.input_pool(x)

            for i, block in enumerate(self.down_blocks, 2):
                x = block(x)
                if i == (self.depth - 1):
                    continue
                pre_pools["layer_"+str(i)] = x

            # print("x",x)
            # bridge = self.bridge(x)
            x = self.avgpool(x)
            x = x.view(x.size(0),-1)
            x = self.down_bottleneck(x)
            # print("bridge",bridge)
            x_reverse = grad_reverse(x, l*-1)
            # print("bridge_reverse",bridge_reverse)
            dom_pred = self.domain_pred(x_reverse)

            # for i, block in enumerate(self.up_blocks, 1):
            #     k = self.depth - 1 - i
            #     key = "layer_"+str(k)
            #     x = block(x, pre_pools[key])
            # output_feature_map = x
            # depth_gen = self.out(x)
            # del pre_pools

            return dom_pred.squeeze()
        # elif train_net:

        #     pre_pools = dict()
        #     pre_pools["layer_0"] = x
        #     x = self.input_block(x)
        #     pre_pools["layer_1"] = x
        #     x = self.input_pool(x)

        #     for i, block in enumerate(self.down_blocks, 2):
        #         x = block(x)
        #         if i == (self.depth - 1):
        #             continue
        #         pre_pools["layer_"+str(i)] = x

        #     x = self.bridge(x)

        #     for i, block in enumerate(self.up_blocks, 1):
        #         k = self.depth - 1 - i
        #         key = "layer_"+str(k)
        #         x = block(x, pre_pools[key])
        #     output_feature_map = x
        #     depth_gen = self.out(x)
        #     del pre_pools

        #     dis_input = torch.cat((x2,depth_gen),0)

        #     dis = self.discriminator_blocks(dis_input).squeeze()
        #     # dis = dis.view(-1, 6*6*512)
        #     # dis = self.discriminate(dis).squeeze()

        #     if with_output_feature_map:
        #         return dis, depth_gen, output_feature_map
        #     else:
        #         return dis, depth_gen

        else:
            return "wrong_mode!"