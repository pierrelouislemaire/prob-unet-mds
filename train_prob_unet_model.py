import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import os
import argparse
import warnings
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
    parser.add_argument('--years_test', type=range, default=range(2080, 2098))
    parser.add_argument('--coords', type=list, default=[120, 184, 120, 184])
    parser.add_argument('--resolution', type=tuple, default=(64, 64))
    parser.add_argument('--lowres_scale', type=int, default=4)
    parser.add_argument('--timetransform', type=str, default='id', choices=['id', 'cyclic'])
    parser.add_argument('--standardization', type=str, default='perpixel', choices=['none', 'perpixel', 'pertimestep', 'minmax'])

    # ML training arguments
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--beta', type=float, default=1.0)
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


def train_probunet_step(model, dataloader, optimizer, epoch, num_epochs, accum, wandb_active, device):

    """
    Performs one epoch of training for the Probabilistic U-Net model.

    Args:
        model (ProbabilisticUNet): The Probabilistic U-Net model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs for training.
        accum (int): Gradient accumulation steps.
        wandb_active (bool): Flag to indicate if Weights & Biases logging is active.
        device (torch.device): Computation device (CPU or GPU).

    Returns:
        float: The mean training loss for the epoch.
    """
        
    model.train()
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(f'Train :: Epoch: {epoch}/{num_epochs}')
        running_losses = []
        for i, batch in enumerate(dataloader):
            tq.update(1)
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            optimizer.zero_grad()
            loss, recon_loss, kl_div = model.elbo(inputs, targets)
            loss.backward()
            optimizer.step()

            if wandb_active:
                wandb.log({
                    "train_loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "kl_div": kl_div.item()
                })

            running_losses.append(loss.item())
            tq.set_postfix_str(s=f'Loss: {loss.item():.4f}')
        
        
        mean_loss = sum(running_losses) / len(running_losses)
        tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')
        return mean_loss

@torch.no_grad()
def eval_probunet_model(model, dataloader, wandb_active, device):

    """
    Evaluates the Probabilistic U-Net model on a validation dataset.

    Args:
        model (ProbabilisticUNet): The Probabilistic U-Net model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        wandb_active (bool): Flag to indicate if Weights & Biases logging is active.
        device (torch.device): Computation device (CPU or GPU).

    Returns:
        float: The mean validation loss.
    """

    model.eval()
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(':: Evaluation ::')
        running_losses = []
        for i, batch in enumerate(dataloader):
            tq.update(1)
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            loss, recon_loss, kl_div = model.elbo(inputs, targets)

            if wandb_active:
                wandb.log({
                    "val_loss": loss.item(),
                    "val_recon_loss": recon_loss.item(),
                    "val_kl_div": kl_div.item()
                })

            running_losses.append(loss.item())

        mean_loss = sum(running_losses) / len(running_losses)
        tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')
        return mean_loss

# In train_prob_unet_model.py

@torch.no_grad()
def sample_probunet_model(model, dataloader, epoch, device):

    """
    Generates and plots samples from the Probabilistic U-Net model.

    Args:
        model (ProbabilisticUNet): The trained Probabilistic U-Net model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        epoch (int): Current epoch number (used for labeling plots).
        device (torch.device): Computation device (CPU or GPU).

    Returns:
        torch.Tensor: The generated high-resolution predictions.
    """
        
    model.eval()
    batch = next(iter(dataloader))
    inputs = batch['inputs'][:2].to(device)  # Select 2 random low-resolution inputs
    lrinterp = batch['lrinterp'][:2]
    hr = batch['hr'][:2]
    timestamps = batch['timestamps'][:2]
    stand_stats = batch['stand_stats'][:2]       

    num_samples = 3  # Number of predicted high-resolution outputs per input
    hr_preds = []

    for _ in range(num_samples):
        output = model(inputs, training=False) # Generate output from the model
        hr_pred = dataloader.dataset.residual_to_hr(output.cpu(), lrinterp, stand_stats)  # Convert residual to high-res
        hr_preds.append(hr_pred) # Append the prediction to the list

    # Stack the predictions along a new dimension to create a tensor of shape [batch_size, num_samples, channels, height, width]
    hr_preds = torch.stack(hr_preds, dim=1)  # Shape: [batch_size, num_samples, channels, height, width]

    # Plot the generated samples using the custom plotting function
    fig, axs = dataloader.dataset.plot_sample_batch(lrinterp, hr_preds, hr, timestamps, epoch, N=2, num_samples=num_samples)

    return hr_preds, (fig, axs)
