import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
from datasets_prep.dataset import create_dataset
from diffusion import sample_from_model, sample_posterior, \
    q_sample_pairs, get_time_schedule, \
    Posterior_Coefficients, Diffusion_Coefficients
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from torch.multiprocessing import Process
from utils import init_processes, copy_source, broadcast_params, to_psnr, compute_psnr_ssim, AverageMeter, get_reliable, \
    update_teachers
from dataset_all import TrainLabeled, TrainUnlabeled, ValLabeled, Train_dataset, Train_undataset
from itertools import cycle
import pyiqa
from loss.losses import *
from loss.contrast import ContrastLoss
from torchvision.models import vgg16
import torch.backends.cudnn as cudnn
from distributed import synchronize
from torch.autograd import Variable
from pytorch_msssim import ssim
from tensorboardX import SummaryWriter
import multiprocessing


def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()

    grad_penalty = args.r1_gamma / 2 * grad_penalty
    grad_penalty.backward()

def train(rank, gpu, args):
    from EMA import EMA
    from score_sde.models.discriminator import Discriminator_large, Discriminator_small
    from score_sde.models.ncsnpp_generator_adagn_pre import NCSNpp, WaveletNCSNpp
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    batch_size = args.batch_size
    nz = args.nz
    train_folder = args.data_dir
    raindrop_dataset = Train_dataset(dataroot=train_folder, phase='labeled', finesize=args.crop_size,
                                     dataset='raindrop')
    rain_dataset = Train_dataset(dataroot=train_folder, phase='labeled', finesize=args.crop_size, dataset='rain')
    snow_dataset = Train_dataset(dataroot=train_folder, phase='labeled', finesize=args.crop_size, dataset='snow')
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    # init smodel and tmodel
    G_NET_ZOO = {"normal": NCSNpp, "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    disc_net = [Discriminator_small, Discriminator_large]
    print("GEN: {}, DISC: {}".format(gen_net, disc_net))

    # init student model
    netG_s = gen_net(args).to(device)
    netD_s = disc_net[0](nc=2 * args.num_channels, ngf=args.ngf,
                         t_emb_dim=args.t_emb_dim,
                         act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)

    # init teacher model
    netG_t = gen_net(args).to(device)
    netD_t = disc_net[0](nc=2 * args.num_channels, ngf=args.ngf,
                         t_emb_dim=args.t_emb_dim,
                         act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)

    broadcast_params(netG_s.parameters())
    broadcast_params(netD_s.parameters())
    broadcast_params(netG_t.parameters())
    broadcast_params(netD_t.parameters())

    optimizerD_s = optim.Adam(filter(lambda p: p.requires_grad, netD_s.parameters(
    )), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG_s = optim.Adam(filter(lambda p: p.requires_grad, netG_s.parameters(
    )), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizerD_t = optim.Adam(filter(lambda p: p.requires_grad, netD_t.parameters(
    )), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG_t = optim.Adam(filter(lambda p: p.requires_grad, netG_t.parameters(
    )), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG_s = EMA(optimizerG_s, ema_decay=args.ema_decay)

    schedulerG_s = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerG_s, args.num_epoch, eta_min=1e-5)
    schedulerD_s = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerD_s, args.num_epoch, eta_min=1e-5)

    # ddp
    netG_s = nn.parallel.DistributedDataParallel(
        netG_s, device_ids=[gpu], find_unused_parameters=True)
    netD_s = nn.parallel.DistributedDataParallel(netD_s, device_ids=[gpu])

    # Wavelet Pooling

    dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
    iwt = DWTInverse(mode='zero', wave='haar').cuda()

    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))
    exp = args.exp
    parent_dir = "./checkpoint"
    log_dir = "./checkpoint_log"
    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models',
                            os.path.join(exp_path, 'score_sde/models'))
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)  #
    global_step, epoch, init_epoch, curiter = 0, 0, 0, 0
    # define loss
    loss_str = MyLoss().to(device, non_blocking=True)
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.cuda()
    perc = PerpetualLoss(vgg_model).cuda()
    max_psnr = -1e18
    max_ssim = -1e18
    for p in netD_t.parameters():
        p.requires_grad = False
    for p in netG_t.parameters():
        p.requires_grad = False
    writer = SummaryWriter(log_dir)

    # def get model output objection
    def get_output(input, label, netG, netD, optimizerG, optimizerD, epoch, is_student=True):
        x = input
        y = label
        x0 = x.to(device, non_blocking=True)
        y0 = y.to(device, non_blocking=True)

        yll, yh = dwt(y0)
        xll, xh = dwt(x0)
        ylh, yhl, yhh = torch.unbind(yh[0], dim=2)
        xlh, xhl, xhh = torch.unbind(xh[0], dim=2)

        real_data = torch.cat([yll, ylh, yhl, yhh], dim=1)
        input_data = torch.cat([xll, xlh, xhl, xhh], dim=1)
        real_data = real_data / 2.0  # [-1, 1]
        assert -1 <= real_data.min() < 0
        assert 0 < real_data.max() <= 1

        if not is_student:
            x_t_1 = torch.randn_like(input_data)
            fake_sample = sample_from_model(
                pos_coeff, netG, args.num_timesteps, x_t_1, input_data, T, args)
            fake_sample = fake_sample * 2
            pred = iwt((fake_sample[:, :3], [torch.stack(
                    (fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
            return pred, y

        # train the netD
        for p in netD.parameters():
            p.requires_grad = True
        netD.zero_grad()
        for p in netG.parameters():
            p.requires_grad = False
        # sample t
        t = torch.randint(0, args.num_timesteps,
                          (real_data.size(0),), device=device)
        x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
        x_t.requires_grad = True
        # train with real
        D_real = netD(x_t, t, x_tp1.detach()).view(-1)
        errD_real = F.softplus(-D_real).mean()
        errD_real.backward(retain_graph=True)
        if args.lazy_reg is None:
            grad_penalty_call(args, D_real, x_t)
        else:
            if global_step % args.lazy_reg == 0:
                grad_penalty_call(args, D_real, x_t)
        # train with fake
        latent_z = torch.randn(batch_size, nz, device=device)
        x_0_predict = netG(torch.cat([x_tp1.detach(), input_data], dim=1), t, latent_z)
        x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)  # get the predict xt-1 an
        output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
        errD_fake = F.softplus(output).mean()
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()
        # update G
        for p in netD.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = True
        netG.zero_grad()
        t = torch.randint(0, args.num_timesteps,
                          (real_data.size(0),), device=device)
        x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
        latent_z = torch.randn(batch_size, nz, device=device)
        x_0_predict = netG(torch.cat([x_tp1.detach(), input_data], dim=1), t, latent_z)
        x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
        output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
        errG = F.softplus(-output).mean()
        #
        x_t_1 = torch.randn_like(real_data)
        fake_sample = sample_from_model(
            pos_coeff, netG, args.num_timesteps, x_t_1, input_data, T, args)
        fake_sample = fake_sample * 2

        pred = iwt((fake_sample[:, :3], [torch.stack(
                (fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
        pred_x0 = iwt((x_0_predict[:, :3], [torch.stack(
                (x_0_predict[:, 3:6], x_0_predict[:, 6:9], x_0_predict[:, 9:12]), dim=2)]))
        str_loss = loss_str(pred_x0, y0)
        per_loss = perc(pred_x0, y0)
        rec_loss = 10 * per_loss + str_loss * 5
        errG = errG + rec_loss
        errG.backward()
        optimizerG.step()
        return pred, y0, errG, errD

    for epoch in range(init_epoch, args.num_epoch):
        if epoch <= args.labeled:
            if epoch<=150:
                paired_dataset=raindrop_dataset
            if epoch>150 and epoch <=300:
                paired_dataset= torch.utils.data.ConcatDataset([raindrop_dataset, rain_dataset])
            if epoch >300:
                paired_dataset = torch.utils.data.ConcatDataset([raindrop_dataset, rain_dataset,snow_dataset])
            pair_sampler = torch.utils.data.distributed.DistributedSampler(paired_dataset,
                                                                           num_replicas=args.world_size,
                                                                           rank=rank)
            pair_loader = torch.utils.data.DataLoader(paired_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=args.num_workers,
                                                      pin_memory=True,
                                                      sampler=pair_sampler,
                                                      drop_last=True)
            paired_loader = iter(pair_loader)
            tbar = range(len(pair_loader))
            tbar = tqdm(tbar, ncols=130, leave=True)
            for i in tbar:
                iteration = i
                img_data, label = next(paired_loader)
                img_data = Variable(img_data).cuda(non_blocking=True)
                label = Variable(label).cuda(non_blocking=True)
                s_pred, s_label, s_errG, s_errD = get_output(img_data, label, netG_s, netD_s, optimizerG_s,
                                                             optimizerD_s,
                                                             epoch, is_student=True)
                writer.add_scalar('s_errG', s_errG, global_step)
                writer.add_scalar('s_errD', s_errD, global_step)
                global_step += 1
                if iteration % 100 == 0:
                    if rank == 0:
                        print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(
                            epoch, iteration, s_errG.item(), s_errD.item()))
            if epoch % 25 == 0:
                content = {
                    'netG_dict': netG_s.state_dict(), 'netD_dict': netD_s.state_dict(),
                }
                torch.save(content, os.path.join(exp_path, 'netG_labeled{}.pth'.format(epoch)))

# %%
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser('ddgan parameters')
    # addf
    parser.add_argument('--data_dir', default='', type=str,
                        help='data root path')
    parser.add_argument('--test_dir', default='', type=str,
                        help='data root path')
    parser.add_argument('--labeled_path', default='./labeled.pth', type=str,
                        help='labeled checkpoint path')
    parser.add_argument('--crop_size', type=int, default=64, help='patcphsize of input.')
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
    parser.add_argument('--beta_max', type=float, default=20,
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

    # add metric thresold
    parser.add_argument('--thresold', type=float, default=0.15,
                        help='whether update candidate.')
    parser.add_argument('--labeled', type=int, default=500,
                        help='train with labeled epoch')

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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_workers')
    # distributed
    # parser.add_argument('--local_rank', type=int, default=0, help="local rank for distributed training")

    args = parser.parse_args()

    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' %
                  (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(
                global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=10)
    else:
        print('starting in debug mode')

        init_processes(0, size, train, args)
