import torch
import torchvision
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import numpy as np
import argparse
from calculate_fid import calculate_mean_std

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path containing either \'cifar-10-batches-py\' and \'celeba\'.')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'celeba', 'lsun_church'],
                        help='Dataset to process statistics,  \'cifar10\' or \'celeba\'. ')
    parser.add_argument('--batch_size', type=int, default=200, 
                        help='Batch size for FID statistics calculation')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device for FID statistics calculation, e.g., \'cuda:0\', \'cuda:1\'.')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        transforms = Compose([
            Resize((32, 32)),
            ToTensor(),
            Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        dataset = torchvision.datasets.CIFAR10(args.data_dir, download=True, transform=transforms)

    elif args.dataset == 'celeba':
        transforms = Compose([
            CenterCrop(140),
            Resize((64, 64)),
            ToTensor(), 
            Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        dataset = torchvision.datasets.CelebA(args.data_dir, download=True, transform=transforms)

    elif args.dataset == 'lsun_church':
        transforms = Compose([
            Resize((128,128)),
            CenterCrop(128),
            ToTensor(), 
            Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        dataset = torchvision.datasets.LSUN(args.data_dir, transform=transforms, classes=['church_outdoor_train'])
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, drop_last=False, num_workers=4)
    
    # calculate mean and covariance of dataset for FID calculations
    full_data = []
    print("Iterating through full dataset...")
    for data in tqdm(dataloader):
        data = data[0]
        full_data.append(data)
    full_data = np.concatenate(full_data, axis=0)

    if args.dataset == 'cifar10' or args.dataset == 'celeba':
        print("Calculating FID statistics of dataset...")
        m, s = calculate_mean_std(full_data, batch_size=args.batch_size, device=args.device, dims=2048, model_type='inception')
        save_file_fid_stats = "eval/" + args.dataset + "_stats.npz"
        print(f"Saving statistics to {save_file_fid_stats}")
        np.savez(save_file_fid_stats, mu=m, sigma=s)
    
    # generate unconditional prior dist
    print("Calculating learned prior ")
    full_data = torch.tensor(full_data)
    full_data_flat = full_data.view(len(full_data), -1)
    mean = full_data_flat.mean(dim=0)
    full_data_flat = full_data_flat - mean.unsqueeze(dim=0)
    cov = full_data_flat.t() @ full_data_flat / len(full_data_flat) # equivalent to torch.cov(full_data_flat.t())
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov+1e-4*torch.eye(len(mean)))
    save_file_prior = args.dataset + "_ddp.pt"
    print(f"Saving learned prior distribution to {save_file_prior}")
    torch.save(dist, save_file_prior)


if __name__ == "__main__":
    main()