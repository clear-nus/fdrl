import torch
import torch.nn as nn
import os
import argparse
import src.utils as utils
import yaml
import os
from src.dataset import toy_datasets
from src.model_utils import flow, get_loss
from src.models import MLP
from src.ema import EMAHelper
import wandb
import time
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help='Path to config file.')
    group.add_argument('--ckpt', type=str, help='Path to checkpoint file.')
    args = parser.parse_args()
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        config = ckpt['config']
    else:
        config = utils.get_config_and_setup_dirs(args.config)
        parser = utils.add_config_to_argparser(config=config, parser=parser)
        args = parser.parse_args()
        # Update config from command line args, if any.
        config.update(vars(args))
    log_dir = config['log_dir']

    if config['use_wandb'] == True:
        wandb.init(project="toy_datasets", config=config, notes=config['wandb_notes'], 
                   id=config['wandb_id'], resume='allow')
        config['run_name'] = wandb.run.name

    if not args.ckpt:
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as fp:
            yaml.dump(config, fp)

    # gpu information
    device = config['device']
    if isinstance(device, list):
        multi_gpu = True
        device_ids = device
        device = torch.device("cuda:" + str(device_ids[0])) # if multi-GPUs, set default device to the first gpu
    else:
        multi_gpu = False
    
    # initialize data
    data_iterator = toy_datasets(config)

    # create model/optimizer/lr scheduler
    model = MLP(config)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=[0.9, 0.999])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, config['scheduler_gamma'])

    # create ema helper
    if config['ema']:
        ema_helper = EMAHelper(mu=config['ema_rate'])
        ema_helper.register(model)

    # load config information
    bs = config['train_batch_size']
    n_epochs = config['n_epochs']
    scheduler_steps = config['scheduler_steps'].copy()
    loss_func = config['loss_function']
    f_divergence = config['f_divergence']
    eta = config['eta']
    noise_factor = config['noise_factor']
    n_flow_steps = config['n_flow_steps']

    # if loading from ckpt, load ckpt
    if args.ckpt:
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optim'])
        ema_helper.load_state_dict(ckpt['ema'])
        ema_helper.to(device) # state_dict from ckpt defaults to cpu
        scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step'] + 1
        while len(scheduler_steps)!=0 and scheduler_steps[0] <= start_step:
            scheduler_steps.pop(0)
    else:
        start_step = 0

    # create test noise for tracking progress of training
    test_noise = torch.randn((bs, 2)).to(device)
    time_start = time.time()
    next_scheduler_step = scheduler_steps.pop(0) if len(scheduler_steps) != 0 else None

    for i in range(start_step, n_epochs):

        x_de = torch.from_numpy(next(data_iterator)).to(device)
        x_nu = torch.randn((bs, 2)).to(device)
        x_nu = flow(x_nu, model, n_flow_steps, eta, noise_factor, f_divergence)

        model_x_nu = model(x_nu)
        model_x_de = model(x_de)
        loss, r_x_nu, r_x_de, first, second = get_loss(loss_func, model_x_nu, model_x_de)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if config['ema']:
            ema_helper.update(model)

        if next_scheduler_step is not None and (i+1) == next_scheduler_step:
            scheduler.step()
            next_scheduler_step = scheduler_steps.pop(0) if len(scheduler_steps) != 0 else None

        if config['use_wandb'] == True:
            wandb.log({"first_term": first,
                    "second_term": second,
                    "loss": loss.item(),
                    "r_x_nu": r_x_nu,
                    "r_x_de": r_x_de
            })

        if (i+1) % config['print_steps'] == 0 or (i+1) == n_epochs:
            print(f"Loss for step {i} is {loss}")
            print(f"R_x_nu is {r_x_nu}, r_x_de is {r_x_de}")
            time_end = time.time()
            print(f"Time taken for steps {i+1-config['print_steps']}-{i} is \
                  {time_end-time_start} seconds\n")
            time_start = time.time()

        if (i+1) % config['log_steps'] == 0 or (i+1) == n_epochs:
            # generate test samples
            test_model = ema_helper.ema_copy(model) if config['ema'] else model
            test_x = test_noise.clone().detach().to(device)
            test_x = flow(test_x, test_model, n_flow_steps, eta, noise_factor, f_divergence).cpu()   
            img_path = os.path.join(log_dir, "step_" + str(i) + ".png")
            f = plt.figure()
            plt.scatter(test_x[:,0], test_x[:,1])
            f.savefig(img_path)

            if config['use_wandb'] == True:
                images = wandb.Image(f, caption=f"Step {i}")
                wandb.log({"generated images": images})

        if (i+1) % config['save_steps'] == 0 or (i+1) == n_epochs:
            torch.save({'step': i,
                        'model': model.state_dict(),
                        'optim': optim.state_dict(),
                        'ema': ema_helper.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'config': config
                        },
                        os.path.join(config['ckpt_dir'], config['dataset'] + '_' + config['loss_function'] + \
                            '_' + config['f_divergence'] + '.pt'))

if __name__ == "__main__":
    main()
