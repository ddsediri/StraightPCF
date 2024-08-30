import os
import time
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchsummary import summary
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from utils.evaluate import *
from models.cvm import *
from models.utils import chamfer_distance_unit_sphere

def input_iter(input_dir):
    for fn in sorted(os.listdir(input_dir)):
        if fn[-3:] != 'xyz':
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {
            'pcl_noisy': pcl_noisy,
            'name': fn[:-4],
            'center': center,
            'scale': scale
        }

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained_cvm/ckpt_cvm.pt')
parser.add_argument('--input_root', type=str, default='./data/examples')
parser.add_argument('--output_root', type=str, default='./data/results')
parser.add_argument('--dataset_root', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='PUNet')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--resolution', type=str, default='10000_poisson')
parser.add_argument('--noise', type=str, default='0.01')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
# Denoiser parameters
parser.add_argument('--patch_size', type=int, default=1000)
parser.add_argument('--seed_k', type=int, default=6)
parser.add_argument('--seed_k_alpha', type=int, default=1)
parser.add_argument('--niters', type=int, default=1)
args = parser.parse_args()
seed_all(args.seed)

# Input/Output
if args.dataset != "RueMadame":
    input_dir = os.path.join(args.input_root, '%s_%s_%s' % (args.dataset, args.resolution, args.noise))
else:
    input_dir = os.path.join(args.input_root, '%s' % (args.dataset))

save_title = '{dataset}_Ours{modeltag}_{tag}_{res}_{noise}_{time}'.format_map({
    'dataset': args.dataset,
    'modeltag': '' if args.niters == 1 else '%dx' % args.niters,
    'tag': args.tag,
    'res': args.resolution,
    'noise': args.noise,
    'time': time.strftime('%m-%d-%H-%M-%S', time.localtime())
})
output_dir = os.path.join(args.output_root, save_title)
os.makedirs(output_dir)
os.makedirs(os.path.join(output_dir, 'pcl'))    # Output point clouds
logger = get_logger('test', output_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = CoupledVMArch(args=ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])
# summar = summary(model)
# logger.info(summar)

# Denoise
for shape_num, data in enumerate(input_iter(input_dir)):
    logger.info("No. {} {}".format(shape_num, data['name']))
    pcl_noisy = data['pcl_noisy'].to(args.device)
    with torch.no_grad():
        model.eval()
        pcl_next = pcl_noisy
        try:
            for niter in range(args.niters):
                pcl_next = patch_based_denoise(
                    model=model,
                    pcl_noisy=pcl_next,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                    patch_size=args.patch_size,
                )
        except Exception as e:
            print("="*100)
            print(e)
            print("="*100)
            print('Current niter is {}'.format(niter))
            print("="*100)

        pcl_denoised = pcl_next.cpu()
        # Denormalize
        pcl_denoised = pcl_denoised * data['scale'] + data['center']
    
    save_path = os.path.join(output_dir, 'pcl', data['name'] + '.xyz')
    np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')

if args.dataset != "RueMadame":
    # Evaluate
    evaluator = Evaluator(
        output_pcl_dir=os.path.join(output_dir, 'pcl'),
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        summary_dir=args.output_root,
        experiment_name=save_title,
        device=args.device,
        res_gts=args.resolution,
        logger=logger
    )
    evaluator.run()
