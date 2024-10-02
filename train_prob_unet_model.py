import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import climex_utils as cu
from prob_unet import ProbabilisticUNet
import trainmodel as tm  
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    num_samples = 3  # Number of predicted high-resolution outputs per input
    hr_preds = []

    for _ in range(num_samples):
        output = model(inputs, training=False) # Generate output from the model
        hr_pred = dataloader.dataset.residual_to_hr(output.cpu(), lrinterp)  # Convert residual to high-res
        hr_preds.append(hr_pred) # Append the prediction to the list

    # Stack the predictions along a new dimension to create a tensor of shape [batch_size, num_samples, channels, height, width]
    hr_preds = torch.stack(hr_preds, dim=1)  # Shape: [batch_size, num_samples, channels, height, width]

    # Plot the generated samples using the custom plotting function
    dataloader.dataset.plot_sample_batch(lrinterp, hr_preds, hr, timestamps, epoch, N=2, num_samples=num_samples)

    return hr_preds





