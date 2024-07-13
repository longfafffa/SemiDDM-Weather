import os
import shutil

import torch
import torch.distributed as dist
from math import log10
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pyiqa
#add
def to_psnr(J, gt):
    mse = F.mse_loss(J, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list



def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        score, diff = structural_similarity(clean[i], recoverd[i], data_range=1, full=True, multichannel=True,
                                            win_size=7, channel_axis=-1)
        ssim = ssim + score

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]
class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
#end add

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(
        backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()
def update_teachers(teacher,student, itera, keep_rate=0.996):
    # exponential moving average(EMA)
    alpha = min(1 - 1 / (itera + 1), keep_rate)
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

def cleanup():
    dist.destroy_process_group()
def get_reliable(teacher_predict, student_predict, positive_list, p_name, score_r,thresold):
    iqa_metric=pyiqa.create_metric('musiq-paq2piq').cuda()

    N =teacher_predict.shape[0]
    score_t = iqa_metric(teacher_predict).detach().cpu().numpy()
    score_s = iqa_metric(student_predict).detach().cpu().numpy()
    positive_sample = positive_list.clone()
    use=0

    for idx in range(0, N):
        if F.l1_loss(teacher_predict[idx], student_predict[idx]) < thresold:
            if score_t[idx] > score_s[idx]:
                if score_t[idx] > score_r[idx]:
                    positive_sample[idx] = teacher_predict[idx]
                    # update the reliable bank
                    temp_c = np.transpose(teacher_predict[idx].detach().cpu().numpy(), (1, 2, 0))
                    temp_c = np.clip(temp_c, 0, 1)
                    arr_c = (temp_c * 255).astype(np.uint8)
                    arr_c = Image.fromarray(arr_c)
                    arr_c.save('%s' % p_name[idx])
                    use+=1

    del N, score_r, score_s, score_t, teacher_predict, student_predict, positive_list
    return positive_sample,use
