import torch.optim as optim
from net.encoder import create_encoder
from net.decoder import create_decoder
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time

parser = argparse.ArgumentParser(description='WITT 4-stage no-channel')
parser.add_argument('--training', action='store_true', default=True,
                    help='training or testing')
parser.add_argument('--test', dest='training', action='store_false',
                    help='testing only')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['kodak', 'CLIC21', 'DIV2K'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--C', type=int, default=32,
                    help='bottleneck dimension')
args = parser.parse_args()

class config():
    seed = 1024
    pass_channel = False
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    tensorboard = workdir + '/tensorboard'
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "/media/Dataset/CIFAR10/"
        test_data_dir = "/media/Dataset/CIFAR10/"
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        train_data_dir = ["/home/lan/data/DIV2K_train_HR/"]
        if args.testset == 'kodak':
            test_data_dir = ["/home/lan/data/Kodak/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/media/Dataset/CLIC21/"]
        elif args.testset == 'DIV2K':
            test_data_dir = ["/home/lan/data/DIV2K_valid_HR/"]
        batch_size = 16
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


class NoChannelWITT(nn.Module):
    def __init__(self, args, config):
        super(NoChannelWITT, self).__init__()
        self.config = config
        self.encoder = create_encoder(**config.encoder_kwargs)
        self.decoder = create_decoder(**config.decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.downsample = config.downsample
        self.H = self.W = 0
        self.model = 'WITT_W/O'

    def forward(self, input_image):
        _, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        feature = self.encoder(input_image, 0, self.model)
        CBR = feature.numel() / 2 / input_image.numel()
        recon_image = self.decoder(feature, 0, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, mse.mean(), loss_G.mean()


def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained


def write_train_scalars(losses, cbrs, psnrs, msssims):
    writer.add_scalar('train/loss', losses.val, global_step)
    writer.add_scalar('train/loss_avg', losses.avg, global_step)
    writer.add_scalar('train/cbr', cbrs.val, global_step)
    writer.add_scalar('train/psnr', psnrs.val, global_step)
    writer.add_scalar('train/ms_ssim', msssims.val, global_step)
    writer.add_scalar('train/lr', cur_lr, global_step)


def write_test_scalars(cbrs, psnrs, msssims):
    writer.add_scalar('test/cbr', cbrs.avg, global_step)
    writer.add_scalar('test/psnr', psnrs.avg, global_step)
    writer.add_scalar('test/ms_ssim', msssims.avg, global_step)


def train_one_epoch(args):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs]
    global global_step
    if args.trainset == 'CIFAR10':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)
            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                write_train_scalars(losses, cbrs, psnrs, msssims)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)

            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                write_train_scalars(losses, cbrs, psnrs, msssims)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()

def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, cbrs = [AverageMeter() for _ in range(4)]
    with torch.no_grad():
        if args.trainset == 'CIFAR10':
            for batch_idx, (input, label) in enumerate(test_loader):
                start_time = time.time()
                input = input.cuda()
                recon_image, CBR, mse, loss_G = net(input)
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnrs.update(100)
                    msssims.update(100)

                log = (' | '.join([
                    f'Time {elapsed.val:.3f}',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
        else:
            for batch_idx, input in enumerate(test_loader):
                start_time = time.time()
                input = input.cuda()
                recon_image, CBR, mse, loss_G = net(input)
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnrs.update(100)
                    msssims.update(100)

                log = (' | '.join([
                    f'Time {elapsed.val:.3f}',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)

    print("CBR: {}".format(cbrs.avg))
    print("PSNR: {}" .format(psnrs.avg))
    print("MS-SSIM: {}".format(msssims.avg))
    write_test_scalars(cbrs, psnrs, msssims)
    print("Finish Test!")

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    writer = SummaryWriter(log_dir=config.tensorboard)
    logger.info('TensorBoard logdir: {}'.format(config.tensorboard))
    torch.manual_seed(seed=config.seed)
    net = NoChannelWITT(args, config)
    # This no-channel architecture matches the 4-stage DIV2K layout in train.py.
    # model_path = "./WITT_model/WITT_WO_AWGN_DIV2K_fixed_snr10_psnr_C96.model"
    # load_weights(model_path)
    net = net.cuda()
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    try:
        if args.training:
            for epoch in range(steps_epoch, config.tot_epoch):
                train_one_epoch(args)
                if (epoch + 1) % config.save_model_freq == 0:
                    save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
                    test()
        else:
            test()
    finally:
        writer.close()

