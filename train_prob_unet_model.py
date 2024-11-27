import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import argparse
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import climex_utils as cu

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
    parser.add_argument('--years_train', type=range, default=range(1960, 1990))
    parser.add_argument('--years_subtrain', type=range, default=range(1960, 1980))
    parser.add_argument('--years_earlystop', type=range, default=range(1980, 1990))
    parser.add_argument('--years_val', type=range, default=range(1990, 1998))
    parser.add_argument('--years_test', type=range, default=range(1998, 2006))
    parser.add_argument('--coords', type=list, default=[80, 208, 100, 228])
    parser.add_argument('--resolution', type=tuple, default=(128, 128))
    parser.add_argument('--lowres_scale', type=int, default=16)
    parser.add_argument('--timetransform', type=str, default='id', choices=['id', 'cyclic'])
    parser.add_argument('--beta', type=int, default=0.4)

    # Downscaling method 
    parser.add_argument('--ds_model', type=str, default='deterministic_unet', choices=['deterministic_unet', 'probabilistic_unet', 'vae', 'linearcnn', 'bcsd'])

    # ML training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--optimizer', type=object, default=torch.optim.AdamW)

    # Model evaluation arguments
    #parser.add_argument('--metric', type=str, default="rmse", choices=["rmse", "crps", "rapsd", "mae"])

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

class EarlyStopper:

    """
    Class for early stopping in the training loop.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model):

        # if the validation loss is lower than the previous minimum, save the model as best model
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            torch.save(model.state_dict(), f"./last_best_model_hr.pt")
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # if the counter is greater than the patience, load the best model and return True to break training
            if self.counter >= self.patience:
                model.load_state_dict(torch.load(f"./last_best_model_hr.pt"))
                return True, model
        return False, model


def train_probunet_step(model, dataloader, optimizer, epoch, num_epochs, device, variables):

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
        float: averaged loss over the dataloader set
    """
        

    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:

        model.train()
        tq.set_description(f'Train :: Epoch: {epoch}/{num_epochs}')

        running_losses_mae = {var: [] for var in variables}
        running_losses_kl = {var: [] for var in variables}
        step_losses = []

        # Initialize the gradient accumulation steps counter
        
        
        for i, batch in enumerate(dataloader):
            tq.update(1)
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            
            loss, recon_loss, kl_div = model.elbo(inputs, targets, timestamps)
            # loss = loss / accum  # Scale the loss for gradient accumulation

            optimizer.zero_grad()  
        
            loss.backward()
            # accelerator.backward(loss)

            optimizer.step()
            # # Update optimizer after `accum` steps
            # if (i + 1) % accum == 0 or (i + 1) == len(dataloader):
            #     optimizer.step()
            #     optimizer.zero_grad()

            # Log losses for each variable
            for idx, var in enumerate(variables):
                var_loss = recon_loss[idx] 
                kl_loss = kl_div[idx]
                running_losses_mae[var].append(var_loss)
                running_losses_kl[var].append(kl_loss)

            step_losses.append(loss.item()) # Store the original unscaled loss
            tq.set_postfix_str(s=f'Loss: {loss.item():.4f}') # Unscaled loss display  
        
        mean_loss = sum(step_losses) / len(step_losses)
        tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')

        running_losses_mae = {var: sum(running_losses_mae[var]) / len(running_losses_mae[var]) for var in variables}
        running_losses_kl = {var: sum(running_losses_kl[var]) / len(running_losses_kl[var]) for var in variables}

        return running_losses_mae, running_losses_kl

@torch.no_grad()
def eval_probunet_model(model, dataloader, reconstruct, device):

    """
    Evaluates the Probabilistic U-Net model on a validation dataset.

    Args:
        model (ProbabilisticUNet): The Probabilistic U-Net model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        wandb_active (bool): Flag to indicate if Weights & Biases logging is active.
        device (torch.device): Computation device (CPU or GPU).

    Returns:
        float: averaged loss over the dataloader set
    """

    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
    
        model.eval()
        tq.set_description(':: Evaluation ::')

        temporal_mae = {var: [] for var in ["pr", "tasmin", "tasmax"]}
        spatial_mae = {var: [] for var in ["pr", "tasmin", "tasmax"]}
        running_losses_mae = {var: [] for var in ["pr", "tasmin", "tasmax"]}
        running_losses_kl = {var: [] for var in ["pr", "tasmin", "tasmax"]}
        step_losses = []

        for i, batch in enumerate(dataloader):

            tq.update(1)
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            lrinterp = batch['lrinterp']
            hr = batch['hr']
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            # # Use Accelerator's mixed precision autocast if provided
            # if accelerator is not None:
            #     with accelerator.autocast():  # Leverage mixed precision
            #         loss, recon_loss, kl_div = model.elbo(inputs, targets)
        
            # Normal full precision (FP32) evaluation

            if not reconstruct:

                loss, recon_loss, kl_div = model.elbo(inputs, targets, timestamps)
                step_losses.append(loss.item())

                # Log losses for each variable
                for idx, var in enumerate(["pr", "tasmin", "tasmax"]):
                    var_loss = recon_loss[idx] 
                    kl_loss = kl_div[idx]
                    running_losses_mae[var].append(var_loss)
                    running_losses_kl[var].append(kl_loss)

            else:

                preds = model(inputs, t=timestamps, training=False)
                preds_hr = dataloader.dataset.residual_to_hr(preds.cpu(), lrinterp)
                preds_hr[:, 0] = cu.softplus(preds_hr[:, 0])
                preds_hr[:, 2] = cu.softplus(preds_hr[:, 2], c=0) + preds_hr[:, 1]
                hr[:, 0] = cu.softplus(hr[:, 0])
                hr[:, 2] = cu.softplus(hr[:, 2], c=0) + hr[:, 1]
                preds_hr[:, 0] = cu.kgm2sTommday(preds_hr[:, 0])
                preds_hr[:, 1:] = cu.KToC(preds_hr[:, 1:])
                hr[:, 0] = cu.kgm2sTommday(hr[:, 0])
                hr[:, 1:] = cu.KToC(hr[:, 1:])

                # log the mae per timestamp
                for i, var in range(["pr", "tasmin", "tasmax"]):
                    temporal_mae[var].append(list(torch.abs(preds_hr[:, i] - hr[:, i]).mean(dim=(-2,-1))))
                    spatial_mae[var].append(torch.abs(preds_hr[:, i] - hr[:, i]).mean(dim=(0)))

        if not reconstruct:

            mean_loss = sum(step_losses) / len(step_losses)
            tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')

            running_losses_mae = {var: sum(running_losses_mae[var]) / len(running_losses_mae[var]) for var in ["pr", "tasmin", "tasmax"]}
            running_losses_kl = {var: sum(running_losses_kl[var]) / len(running_losses_kl[var]) for var in ["pr", "tasmin", "tasmax"]}

            return running_losses_mae, running_losses_kl
        
        else: 

            temporal_mae = {var: sum(temporal_mae[var], []) for var in ["pr", "tasmin", "tasmax"]}
            spatial_mae = {var: np.sum(spatial_mae[var], axis=0) / len(spatial_mae[var]) for var in ["pr", "tasmin", "tasmax"]}

            return temporal_mae, spatial_mae
    

# @torch.no_grad()
# def plot_losses(train_losses, val_losses, variables, plotdir):
#     """
#     Plots the training and validation losses for each variable and saves the figure.
#     """
#     for var in variables:
#         plt.figure(figsize=(10, 6))
#         plt.plot(train_losses[var], label='Training', lw=2)
#         plt.plot(val_losses[var], label='Validation', lw=2, linestyle='--')
#         plt.title(f'Training and Validation Losses for {var}')
#         plt.xlabel('Steps')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.savefig(f'{plotdir}/{var}_loss.png', dpi=300)
#         plt.close()
    

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
    lrinterp = batch['lrinterp'][:2].to(device)
    hr = batch['hr'][:2].to(device)
    # Ensure timestamps has shape (batch_size, 1)
    timestamps = batch['timestamps'][:2].unsqueeze(dim=1).to(device)
    timestamps_float = batch['timestamps_float'][:2]  # For plotting
   

    num_samples = 3  # Number of predicted high-resolution outputs per input
    hr_preds = []

    for _ in range(num_samples):
        output = model(inputs, t=timestamps, training=False) # Generate output from the model
        hr_pred = dataloader.dataset.residual_to_hr(output.cpu(), lrinterp.cpu())  # Convert residual to high-res
        hr_preds.append(hr_pred) # Append the prediction to the list

    # Stack the predictions along a new dimension to create a tensor of shape [batch_size, num_samples, channels, height, width]
    hr_preds = torch.stack(hr_preds, dim=1)  # Shape: [batch_size, num_samples, channels, height, width]

    # print(f"timestamps shape: {timestamps.shape}")  
    # print(f"timestamps value: {timestamps}") 
    # print(f"timestamps_float: {timestamps_float}")
    # print(f"timestamps_float shape: {timestamps_float.shape}")

    # Plot the generated samples using the custom plotting function
    fig, axs = dataloader.dataset.plot_sample_batch(lrinterp.cpu(), hr_preds.cpu(), hr.cpu(), timestamps_float, epoch, N=2, num_samples=num_samples)

    return hr_preds, (fig, axs)