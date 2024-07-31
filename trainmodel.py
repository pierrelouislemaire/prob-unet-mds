import os
import warnings
import argparse
import torch
import wandb

from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

def get_args():

    """
    This function returns a dictionary containing all necessary arguments for importing ClimEx data, training, evaluating and sampling from a downscaling ML model.
    This function is helpful for doing sweeps and performing hyperparameter tuning.
    """

    parser = argparse.ArgumentParser()

    # climate dataset arguments
    parser.add_argument('--datadir', type=str, default='/home/julie/Data/Climex/day/kdj/')
    parser.add_argument('--variables', type=list, default=['pr', 'tasmin', 'tasmax'])
    parser.add_argument('--years_train', type=range, default=range(1960, 2060))
    parser.add_argument('--years_val', type=range, default=range(2060, 2080))
    parser.add_argument('--years_test', type=range, default=range(2080, 2099))
    parser.add_argument('--coords', type=list, default=[120, 184, 120, 184])
    parser.add_argument('--resolution', type=tuple, default=(64, 64))
    parser.add_argument('--lowres_scale', type=int, default=4)
    parser.add_argument('--timetransform', type=str, default='id', choices=['id', 'cyclic'])

    # ML training arguments
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--optimizer', type=object, default=torch.optim.AdamW)

    # WandB activation 
    parser.add_argument('--wandb', type=bool, default=False)

    # GPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # saving results arguments
    strtime = datetime.now().strftime('%m/%d/%Y%H:%M:%S')
    plotdir = './results/plots/' + strtime + '/'
    parser.add_argument('--plotdir', type=str, default=plotdir)
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    # Generating the dictionary
    args, unknown = parser.parse_known_args()

    return args


def train_step(model, dataloader, loss_fn, optimizer, scaler, epoch, num_epochs, accum, wandb, device):

    """
    Function for training the UNet model for a single epoch.

    model: instance of the Unet class
    dataloader: torch training dataloader
    loss_fn: loss function
    optimizer: torch optimizer 
    scaler: scaler for mixed precision training
    epoch: current epoch
    num_epochs: total number of epochs
    accum: number of steps to accumulate gradients over
    wandb: True if wandb activated
    device: device to use (GPU)

    return -> average loss value over the epoch
    """

    model.train()

    # Activating progress bar
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(f'Train :: Epoch: {epoch}/{num_epochs}')

        running_losses = []
        # Looping over the entire dataloader set
        for i, batch in enumerate(dataloader):
            tq.update(1)

            # Extracting training data from batch
            inputs, targets = (batch['inputs'].to(device), batch['targets'].to(device))
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            # Performing forward pass of the unet model qnd computing loss
            with torch.cuda.amp.autocast():
                preds = model(inputs, class_labels=timestamps)
                loss = loss_fn(preds, targets)

            # Logging training loss in wandb
            if wandb:
                wandb.log(data={"train-loss": loss.item()})

            # Performing backprograpation
            scaler.scale(loss).backward()

            # Updating optimizer every accum steps
            if (i + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_losses.append(loss.item())
            tq.set_postfix_str(s=f'Loss: {loss.item():.4f}')

        mean_loss = sum(running_losses) / len(running_losses)
        tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')

        return mean_loss

@torch.no_grad()
def sample_model(model, dataloader, epoch, device):

    """
    Function for sampling from the unet model and plotting results.

    model: instance of the Unet class
    dataloader: torch dataloader (should be shuffled)
    epoch: last training epoch
    device: device to use (GPU)

    return -> predicted high-resolution samples, plots
    """

    model.eval()

    # Extracting data from the first batch of the dataloader
    batch = next(iter(dataloader))
    inputs, lrinterp, hr, timestamps = (batch['inputs'].to(device), batch['lrinterp'], batch['hr'], batch['timestamps'])

    # Performing forward pass on the unet
    residual_preds = model(inputs, class_labels=timestamps.unsqueeze(dim=1).to(device))

    # Converting predicted residual to high-resolution sample
    hr_pred = dataloader.dataset.residual_to_hr(residual_preds.detach().cpu(), lrinterp)

    # Plotting the results
    fig, axs = dataloader.dataset.plot_batch(lrinterp, hr_pred, hr, timestamps, epoch)

    return hr_pred, (fig, axs)

@torch.no_grad()
def eval_model(model, dataloader, loss_fn, wandb, device):

    """
    Function for evaluating the unet model.

    model: instance of the Unet class
    dataloader: torch dataloader 
    loss_fn: loss function used for evaluation
    epoch: last training epoch
    wandb: True if wandb activated
    device: device to use (GPU)

    return -> averaged loss over the dataloader set
    """

    model.eval()

    # Activating progress bar
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(':: Evaluation ::')

        running_losses = []
        # Looping over the entire dataloader set
        for i, batch in enumerate(dataloader):
            tq.update(1)

            # Extracting training data from batch and performing forward pass
            inputs, targets, timestamps = (batch['inputs'].to(device), batch['targets'].to(device), batch['timestamps'])
            residual_preds = model(inputs, class_labels=timestamps.unsqueeze(dim=1).to(device))

            # Computing loss function
            loss = loss_fn(residual_preds, targets)
            running_losses.append(loss.item())

            if wandb:
                wandb.log(data={"val-loss": loss.item()})

        mean_loss = sum(running_losses) / len(running_losses)
        tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')

    return mean_loss
