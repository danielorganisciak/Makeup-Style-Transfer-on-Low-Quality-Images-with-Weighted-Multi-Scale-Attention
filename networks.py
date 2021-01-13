import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from modules import ResnetBlock, CondInstanceNorm, TwoInputSequential, CINResnetBlock, InstanceNorm2d

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('ConvBlock') != -1:
        m.conv.weight.data.normal_(0.0, 0.02)
        m.bn.weight.data.normal_(1.0, 0.02)
        m.bn.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(InstanceNorm2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, norm='instance', which_model_netG='resnet',
             use_dropout=False, gpu_ids=[]):

    netG = None
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())

    norm_layer = get_norm_layer(norm_type=norm)

    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                           use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG


def define_stochastic_G(nlatent, input_nc, output_nc, ngf, norm='instance',
                        which_model_netG='resnet', use_dropout=False, gpu_ids=[]):

    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    norm_layer = CondInstanceNorm

    netG = CINResnetGenerator(nlatent, input_nc, output_nc, ngf, norm_layer=norm_layer,
                              use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG


def define_D_A(input_nc, ndf, which_model_netD, norm, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = Discriminator_edges(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def define_D_B(input_nc, ndf, which_model_netD, norm, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = Discriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD

def define_LAT_D(nlatent, ndf, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = DiscriminatorLatent(nlatent, ndf, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD

def define_E(nlatent, input_nc, nef, norm='batch', gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    netE = LatentEncoder(nlatent, input_nc, nef, norm_layer=norm_layer, gpu_ids=gpu_ids)

    if use_gpu:
        netE.cuda()
    netE.apply(weights_init)
    return netE


def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    if out_f is not None:
        out_f.write(net.__repr__()+"\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()

##########################################
#               Attention
##########################################

class ConvBlock(nn.Module):
    """Basic convolutional block.

    convolution + batch normalization + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(
            x, (x.size(2) * 2, x.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        # scaling conv
        x = self.conv2(x)
        return x


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()

        if in_channels == 6:
            reduction_rate = 6
        #print(in_channels, reduction_rate, in_channels % reduction_rate)
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)

    Aim: Spatial Attention + Channel Attention

    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)
        self.in_channels = in_channels

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        #if x.size(2) != y.size(2):
        #    x = F.upsample(
        #        x, (y.size(2), y.size(3)),
        #        mode='bilinear',
        #        align_corners=True
        #    )
        x = x * y
        return x


class MultiscaleSpatialAttn(nn.Module):

    def __init__(self):
        super(MultiscaleSpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)
        self.conv5 = ConvBlock(1, 1, 5, s=1, p=2)
        self.channel_attn = ChannelAttn(3, 3)
        self.conv3 = ConvBlock(3, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv5(x)
        # bilinear resizing
        x = F.upsample(
            x, (x.size(2) * 2, x.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        x2 = F.upsample(
            x2, (x2.size(2) * 2, x2.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        x3 = F.upsample(
            x3, (x3.size(2) * 2, x3.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        # scaling conv
        x = torch.cat((x, x2, x3), 1)
        y = self.channel_attn(x)
        x = x*y
        x = self.conv3(x)
        return x

class MultiscaleSoftAttn(nn.Module):


    def __init__(self, in_channels):
        super(MultiscaleSoftAttn, self).__init__()
        self.spatial_attn = MultiscaleSpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)
        self.in_channels = in_channels

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        #if x.size(2) != y.size(2):
        #    x = F.upsample(
        #        x, (y.size(2), y.size(3)),
        #        mode='bilinear',
        #        align_corners=True
        #    )
        x = x * y
        return x


##############################################################################
# Network Classes
##############################################################################

######################################################################
# Modified version of ResnetGenerator that supports stochastic mappings
# using Conditonal instance norm (can support CBN easily)
######################################################################
class CINResnetGenerator(nn.Module):
    def __init__(self, nlatent, input_nc, output_nc, ngf=64, norm_layer=CondInstanceNorm,
                 use_dropout=False, n_blocks=9, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(CINResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        instance_norm = functools.partial(InstanceNorm2d, affine=True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2*ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2*ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4*ngf, nlatent),
            nn.ReLU(True)
        ]
        
        for i in range(3):
            model += [CINResnetBlock(x_dim=4*ngf, z_dim=nlatent, padding_type=padding_type,
                                     norm_layer=norm_layer, use_dropout=use_dropout, use_bias=True)]

        model += [
            nn.ConvTranspose2d(4*ngf, 2*ngf, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=True),
            norm_layer(2*ngf, nlatent),
            nn.ReLU(True),

            MultiscaleSoftAttn(2*ngf),

            nn.Conv2d(2*ngf, ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            MultiscaleSoftAttn(ngf),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = TwoInputSequential(*model)

    def forward(self, input, noise):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, noise), self.gpu_ids)
        else:
            return self.model(input, noise)


######################################################################
# ResnetGenerator for deterministic mappings
######################################################################
class CINResnetGenerator3(nn.Module):
    def __init__(self, nlatent, input_nc, output_nc, ngf=64, norm_layer=InstanceNorm2d, use_dropout=False,
                 n_blocks=9, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(CINResnetGenerator3, self).__init__()
        self.gpu_ids = gpu_ids

        instance_norm = InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            instance_norm(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2*ngf, kernel_size=3, padding=1, stride=1, bias=True),
            instance_norm(2*ngf),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, padding=1, stride=2, bias=True),
            instance_norm(4*ngf),
            nn.ReLU(True)
        ]

        for i in range(3):
            model += [CINResnetBlock(x_dim=4*ngf, z_dim=nlatent, padding_type=padding_type,
                                     norm_layer=norm_layer, use_dropout=use_dropout, use_bias=True)]

        model += [
            nn.ConvTranspose2d(4*ngf, 2*ngf, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=True),
            norm_layer(2*ngf , nlatent),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            MultiscaleSoftAttn(ngf),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = TwoInputSequential(*model)


    def forward(self, input, noise):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, noise), self.gpu_ids)
        else:
            return self.model(input, noise)




######################################################################
# ResnetGenerator for deterministic mappings
######################################################################
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=InstanceNorm2d, use_dropout=False,
                 n_blocks=9, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2*ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2*ngf),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4*ngf),
            nn.ReLU(True),
        ]

        for i in range(3):
            model += [ResnetBlock(4*ngf, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=True)]

        model += [

            nn.ConvTranspose2d(4*ngf, 2*ngf,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=True),
            norm_layer(2*ngf),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, ngf, kernel_size=3, padding=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            MultiscaleSoftAttn(ngf),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


######################################################################
# Discriminator that supports stochastic mappings
# using Conditonal instance norm (can support CBN easily)
######################################################################
class CINDiscriminator(nn.Module):
    def __init__(self, nlatent, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, gpu_ids=[]):
        """
        nlatent: number of channels in both latent codes (or one of them - depending on the model)
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(CINDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.att_model = SoftAttn

        use_bias = True

        kw = 4
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),

            MultiscaleSoftAttn(ndf),

            nn.Conv2d(ndf, 2*ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2*ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            #SoftAttn(2*ndf),

            nn.Conv2d(2*ndf, 4*ndf,
                      kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(4*ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4*ndf, 5*ndf,
                      kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(5*ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(5*ndf, 1, kernel_size=kw, stride=1, padding=1)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = TwoInputSequential(*sequence)

    def forward(self, input, noise):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, noise), self.gpu_ids)
        else:
            return self.model(input, noise)


######################################################################
# Discriminator network
######################################################################
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, gpu_ids=[]):
        """
        nlatent: number of channels in both latent codes (or one of them - depending on the model)
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids

        use_bias = True

        kw = 4
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),

            MultiscaleSoftAttn(ndf),

            nn.Conv2d(ndf, 2*ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2*ndf),
            nn.LeakyReLU(0.2, True),

            #SoftAttn(2*ndf),

            nn.Conv2d(2*ndf, 4*ndf, kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(4*ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4*ndf, 4*ndf, kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(4*ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4*ndf, 1, kernel_size=kw, stride=1, padding=1)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class Discriminator_edges(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, gpu_ids=[]):
        """
        nlatent: number of channles in both latent codes (or one of them - depending on the model)
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(Discriminator_edges, self).__init__()
        self.gpu_ids = gpu_ids

        use_bias = True

        kw = 3
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, 2*ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2*ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(2*ndf, 4*ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(4*ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4*ndf, 4*ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(4*ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4*ndf, 1, kernel_size=4, stride=1, padding=0, bias=True)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class DiscriminatorLatent(nn.Module):
    def __init__(self, nlatent, ndf,
                 use_sigmoid=False, gpu_ids=[]):
        super(DiscriminatorLatent, self).__init__()

        self.gpu_ids = gpu_ids
        self.nlatent = nlatent

        use_bias = True
        sequence = [
            nn.Linear(nlatent, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, 1)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if input.dim() == 4:
            input = input.view(input.size(0), self.nlatent)

        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

######################################################################
# Encoder network for latent variables
######################################################################
class LatentEncoder(nn.Module):
    def __init__(self, nlatent, input_nc, nef, norm_layer, gpu_ids=[]):
        super(LatentEncoder, self).__init__()
        self.gpu_ids = gpu_ids
        use_bias = False

        kw = 3
        sequence = [

            MultiscaleSoftAttn(input_nc),

            nn.Conv2d(input_nc, nef, kernel_size=kw, stride=2, padding=1, bias=True),
            nn.ReLU(True),

            MultiscaleSoftAttn(1*nef),

            nn.Conv2d(nef, 2*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2*nef),
            nn.ReLU(True),

            MultiscaleSoftAttn(2*nef),

            nn.Conv2d(2*nef, 4*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(4*nef),
            nn.ReLU(True),

            nn.Conv2d(4*nef, 8*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(8*nef),
            nn.ReLU(True),

            nn.Conv2d(8*nef, 8*nef, kernel_size=8, stride=1, padding=0, bias=use_bias),
            norm_layer(8*nef),
            nn.ReLU(True),

        ]

        self.conv_modules = nn.Sequential(*sequence)

        # make sure we return mu and logvar for latent code normal distribution
        self.enc_mu = nn.Conv2d(8*nef, nlatent, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc_logvar = nn.Conv2d(8*nef, nlatent, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            #if input.size()[0] != 20:
            #    print (input.size())
            conv_out = nn.parallel.data_parallel(self.conv_modules, input, self.gpu_ids)
            mu = nn.parallel.data_parallel(self.enc_mu, conv_out, self.gpu_ids)
            logvar = nn.parallel.data_parallel(self.enc_logvar, conv_out, self.gpu_ids)
        else:
            conv_out = self.conv_modules(input)
            mu = self.enc_mu(conv_out)
            logvar = self.enc_logvar(conv_out)
        return (mu.view(mu.size(0), -1), logvar.view(logvar.size(0), -1))

