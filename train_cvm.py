import os
import shutil
import argparse
import torch
from torch import nn
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchsummary import summary
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from models.vm import *
from models.cvm import *
from models.utils import chamfer_distance_unit_sphere


# Arguments
parser = argparse.ArgumentParser()
## Dataset and loader
parser.add_argument('--velocity_ckpt', type=str, default='./pretrained_vm/ckpt_vm.pt')
parser.add_argument('--dataset_root', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='PUNet')
parser.add_argument('--patch_size', type=int, default=1000)
parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'])
parser.add_argument('--noise_min', type=float, default=0.005) #0.005
parser.add_argument('--noise_max', type=float, default=0.020)
parser.add_argument('--train_batch_size', type=int, default=8)
# parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
## Model architecture
parser.add_argument('--supervised', type=eval, default=True, choices=[True, False])
parser.add_argument('--tot_its', type=int, default=10)
parser.add_argument('--frame_knn', type=int, default=32)
parser.add_argument('--num_train_points', type=int, default=128)
parser.add_argument('--dsm_sigma', type=float, default=0.01)
parser.add_argument('--feat_embedding_dim', type=int, default=256)
parser.add_argument('--decoder_hidden_dim', type=int, default=64)
## Optimizer and scheduler
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
## Training
parser.add_argument('--train_cvm_network', type=eval, default=False, choices=[True, False])
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--num_modules', type=int, default=4)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=1*MILLION)
parser.add_argument('--val_freq', type=int, default=2000)
parser.add_argument('--val_noise', type=float, default=0.015) # 0.015
parser.add_argument('--val_num_visualize', type=int, default=4)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

if args.optimizer == 'SGD':
    args.lr = 1e-4

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='CVM_D%s_' % (args.dataset), postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
    model_src = './models'
    utils_src = './utils'
    datasets_src = './datasets'
    shutil.copy('./train_cvm.py', log_dir) 
    shutil.copy('./test_cvm.py', log_dir) 
    shutil.copytree(model_src, os.path.join(log_dir, 'models')) 
    shutil.copytree(utils_src, os.path.join(log_dir, 'utils'))
    shutil.copytree(datasets_src, os.path.join(log_dir, 'datasets')) 
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets')
train_dset = PairedPatchDataset(
    datasets=[
        PointCloudDataset(
            root=args.dataset_root,
            dataset=args.dataset,
            split='train',
            resolution=resl,
            transform=standard_train_transforms(noise_std_max=args.noise_max, noise_std_min=args.noise_min, rotate=args.aug_rotate)
        ) for resl in args.resolutions
    ],
    patch_size=args.patch_size,
    patch_ratio=1.2,
    train_cvm_network=args.train_cvm_network,
    tot_its=args.tot_its,
)
val_dset = PointCloudDataset(
        root=args.dataset_root,
        dataset=args.dataset,
        split='test',
        resolution=args.resolutions[0],
        transform=standard_train_transforms(noise_std_max=args.val_noise, noise_std_min=args.val_noise, rotate=False, scale_d=0),
    )
train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True))

# Model
logger.info('Building model...')
velocity_ckpt = torch.load(args.velocity_ckpt, map_location=args.device)
velocity_models = nn.ModuleList()
for i in range(args.num_modules):
    velocity_models.append(VelocityModule(args=velocity_ckpt['args']).to(args.device))
    velocity_models[i].load_state_dict(velocity_ckpt['state_dict'])
model = CoupledVMArch(velocity_models, args).to(args.device)
# logger.info(repr(model))
logger.info(summary(model))

# Optimizer and scheduler
if args.optimizer == 'Adam':
    logger.info('Using Adam optimizer')
    optimizer = torch.optim.Adam(model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
elif args.optimizer == 'SGD':
    logger.info('Using SGD optimizer')
    optimizer = torch.optim.SGD(model.parameters(),
        lr=args.lr,
        momentum=0.9,
    )

# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)
    pcl_clean = batch['pcl_clean'].to(args.device)
    pcl_noisy_L2 = batch['pcl_noisy_L2'].to(args.device)
    pcl_seeds_t = batch['seed_points_t'].to(args.device)
    original_time_step = batch['original_time_step'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    full_loss = model.get_supervised_loss(pcl_clean=pcl_clean,
                                          pcl_noisy_L2=pcl_noisy_L2,
                                          pcl_seeds_t=pcl_seeds_t, 
                                          original_time_step=original_time_step)

    # Backward and optimize
    full_loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    # Logging
    logger.info('[Train] Iter %04d | Full loss %.6f | Grad %.6f' % (
        it, full_loss.item(), orig_grad_norm,
    ))
    writer.add_scalar('train/loss', full_loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush() 

def validate(it):
    all_clean = []
    all_denoised = []
    for i, data in enumerate(tqdm(val_dset, desc='Validate')):
        pcl_noisy = data['pcl_noisy'].to(args.device)
        pcl_clean = data['pcl_clean'].to(args.device)
        pcl_std = torch.tensor(data['noise_std'], dtype=torch.float).to(args.device)
        pcl_denoised = patch_based_denoise(model, pcl_noisy)
        all_clean.append(pcl_clean.unsqueeze(0))
        all_denoised.append(pcl_denoised.unsqueeze(0))
    all_clean = torch.cat(all_clean, dim=0)
    all_denoised = torch.cat(all_denoised, dim=0)

    avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()

    logger.info('[Val] Iter %04d | CD %.6f  ' % (it, avg_chamfer))
    writer.add_scalar('val/chamfer', avg_chamfer, it)
    writer.add_mesh('val/pcl', all_denoised[:args.val_num_visualize], global_step=it)
    writer.flush()

    # scheduler.step(avg_chamfer)
    return avg_chamfer

# Main loop
logger.info('Start training...')
try:
    for it in range(1, args.max_iters+1):
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            cd_loss = validate(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
            # ckpt_mgr.save(model, args, 0, opt_states, step=it)

except KeyboardInterrupt:
    logger.info('Terminating...')
