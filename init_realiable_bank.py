
import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasets_prep.dataset import create_dataset
from diffusion import sample_from_model, sample_posterior, \
    q_sample_pairs, get_time_schedule, \
    Posterior_Coefficients, Diffusion_Coefficients
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from torch.multiprocessing import Process
from utils import init_processes, copy_source, broadcast_params,to_psnr,compute_psnr_ssim,AverageMeter
from dataset_all import Val_dataset
from collections import OrderedDict
from loss.losses import *
from torchvision.models import vgg16
import torch.backends.cudnn as cudnn
from distributed import synchronize

def sample_and_test(args,device):
    from EMA import EMA
    from score_sde.models.discriminator import Discriminator_large, Discriminator_small
    from score_sde.models.ncsnpp_generator_adagn_pre import NCSNpp, WaveletNCSNpp
    batch_size = args.batch_size
    rank=0
    nz = args.nz  # latent dimension
    #get val_dataset
    train_folder = args.data_dir
    val_dataset=Val_dataset(dataroot=train_folder,finesize=args.crop_size)
    val_sampler= torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                   num_replicas=args.world_size,
                                                                   rank=rank)
    val_loader=torch.utils.data.DataLoader(val_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              sampler=None,
                                              drop_last=False)

    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    netG = gen_net(args).to(device)

    # Wavelet Pooling
    if not args.use_pytorch_wavelet:
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
        iwt = DWTInverse(mode='zero', wave='haar').cuda()

    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))
    #set save_path
    exp = args.exp
    parent_dir = "./saved_info/wdd_gan/{}".format(args.dataset)

    ckpt = torch.load(args.path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in ckpt['netG_dict'].items()}
    netG.load_state_dict(new_state_dict, strict=True)
    netG.eval()

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)  #

    for iteration, (data, path) in enumerate(val_loader):
        x0 = data.to(device, non_blocking=True)

        xll, xh = dwt(x0)
        xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
        input_data = torch.cat([xll, xlh, xhl, xhh], dim=1)

        x_t_1 = torch.randn_like(input_data)
        fake_sample = sample_from_model(
            pos_coeff, netG, args.num_timesteps, x_t_1, input_data, T, args)
        x_0_predict=fake_sample*2.0
        val_pred = iwt((x_0_predict[:, :3], [torch.stack(
            (x_0_predict[:, 3:6], x_0_predict[:, 6:9], x_0_predict[:, 9:12]), dim=2)]))
        torchvision.utils.save_image(val_pred, path[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
# add
    parser.add_argument('--data_dir', default='', type=str,
                    help='data root path')
    parser.add_argument('--crop_size', type=int, default=64, help='patcphsize of input.')
    parser.add_argument('--path', default='', type=str,
                        help='the first stage training weight')
    parser.add_argument('--seed', type=int, default=1024,
                    help='seed used for initialization')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--image_size', type=int, default=128,
                    help='size of image')
    parser.add_argument('--num_channels', type=int, default=12,
                    help='channel of wavelet subbands')
    parser.add_argument('--centered', action='store_false', default=True,
                    help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                    help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                    help='beta_max for diffusion')
    parser.add_argument('--patch_size', type=int, default=1,
                    help='Patchify image into non-overlapped patches')
    parser.add_argument('--num_channels_dae', type=int, default=64,
                    help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=4,
                    help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                    help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                    help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), nargs='+', type=int,
                    help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                    help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                    help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                    help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                    help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                    help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                    help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                    help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                    help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                    help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                    help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                    help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                    help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
# generator and training
    parser.add_argument(
        '--exp', default='wddgan_cifar10_exp1_noatn_g122_d3_recloss_1800ep', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data/cifar-10')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int,
                    default=1, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=900)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float,
                    default=1.6e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1.25e-4,
                    help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                    help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=True,
                    help='use EMA or not')
    parser.add_argument('--ema_decay', type=float,
                    default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float,
                    default=0.02, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=15,
                    help='lazy regulariation.')

# wavelet GAN
    parser.add_argument("--current_resolution", type=int, default=128)
    parser.add_argument("--use_pytorch_wavelet", action="store_true", default=True)
    parser.add_argument("--rec_loss", action="store_true", default=True)
    parser.add_argument("--net_type", default="wavelet")
    parser.add_argument("--num_disc_layers", default=5, type=int)
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")

    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=10,
                    help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int,
                        default=25, help='save ckpt every x epochs')

# ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                    help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=2,
                    help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                    help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=1,
                    help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                    help='address for master')
    parser.add_argument('--master_port', type=str, default='6036',
                    help='port for master')
    parser.add_argument('--num_workers', type=int, default=1,
                    help='num_workers')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node



    sample_and_test(args,device)
