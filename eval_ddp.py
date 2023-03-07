import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from src.models import get_model
from src.ema import EMAHelper
from src.model_utils import flow
import argparse
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from calculate_fid import calculate_fid_given_arr_and_stats

# prevents some machines from freezing when calculating FID
# due to scipy.linalg.sqrtm calculation
# see https://github.com/scipy/scipy/issues/14594
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, type=str,
                        help='path to .pt file of model being evaluated')
    parser.add_argument('--batch_size', type=int, default=200, 
                        help='Batch size for image generation and FID calculation')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device for image generation and FID calculation e.g., \'cuda:0\', \'cuda:1\'.')
    parser.add_argument('--n_flow_steps', type=int, default=120,
                        help='Total number of flow steps')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    config = ckpt['config']

    device = args.device
    config['device'] = device
    model = get_model(config)
    
    if config['ema']:
        ema_helper = EMAHelper(mu=config['ema_rate'])
        ema_helper.load_state_dict(ckpt['ema'])
        model = ema_helper.ema_copy(model)
    else:
        model.load_state_dict(ckpt['model'])
    model = model.to(device)
    print(f"Loading checkpoint step {ckpt['step']} for dataset {config['dataset']}")

    ddp = torch.load(config['dataset'] + "_ddp.pt")
    os.makedirs("samples", exist_ok=True)

    n_fid = 50000
    fid_batch = ddp.sample([n_fid]).reshape(-1,3, 
                        config['img_size'], config['img_size']).clamp(-1.,1.).to(device)
    fid_ds = TensorDataset(fid_batch)
    fid_dl = DataLoader(fid_ds, args.batch_size)

    print("Generating 50k images for FID calculation...")
    arr = []
    for _, data in enumerate(tqdm(fid_dl)):
        fid_batch = data[0].to(device)
        fid_batch = flow(fid_batch, model, args.n_flow_steps, config['eta'], 
                                        config['noise_factor'], config['f_divergence'])
        fid_batch = fid_batch.clamp(-1.,1.)
        arr.append(fid_batch.cpu().numpy())
    arr = np.concatenate(arr)

    save_img = torch.tensor(arr[:400])
    grid = make_grid(save_img.clamp(-1., 1.), nrow=20, normalize=True, padding=1)
    f = plt.figure(figsize=(20,20))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    save_file = "samples/" + str(config['dataset']) + "_" + str(config["loss_function"]) + \
                "_" + str(config['f_divergence']) + ".png"
    print(f"Saving test samples to {save_file}")
    f.savefig(save_file)

    # get dataset statistics for fid calculations
    with np.load('eval/' + str(config['dataset']) + "_stats.npz") as f:
        true_m, true_s = f['mu'][:], f['sigma'][:]
    fid = calculate_fid_given_arr_and_stats(arr, true_m, true_s, device=device)
    print(f'FID for {config["dataset"]}-{config["loss_function"]}-{config["f_divergence"]}: {fid}\n')

if __name__ == "__main__":
    main()