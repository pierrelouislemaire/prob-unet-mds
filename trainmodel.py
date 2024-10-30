import os
import warnings
import argparse
import torch
import wandb
import metrics
import numpy as np

import climex_utils as cu 

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
    parser.add_argument('--years_train', type=range, default=range(1990, 2020))
    parser.add_argument('--years_val', type=range, default=range(2020, 2025))
    parser.add_argument('--years_test', type=range, default=range(2025, 2030))
    parser.add_argument('--coords', type=list, default=[80, 208, 100, 228])
    parser.add_argument('--resolution', type=tuple, default=(128, 128))
    parser.add_argument('--lowres_scale', type=int, default=4)
    parser.add_argument('--timetransform', type=str, default='id', choices=['id', 'cyclic'])

    # Downscaling method 
    parser.add_argument('--ds_model', type=str, default='deterministic_unet', choices=['deterministic_unet', 'probabilistic_unet', 'vae', 'linearcnn', 'bcsd'])

    # ML training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--optimizer', type=object, default=torch.optim.AdamW)

    # Model evaluation arguments
    #parser.add_argument('--metric', type=str, default="rmse", choices=["rmse", "crps", "rapsd", "mae"])

    # WandB activation 
    parser.add_argument('--wandb', type=bool, default=False)

    # GPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    # saving results arguments
    strtime = datetime.now().strftime('%m/%d/%Y%H')
    plotdir = './results/plots/' + strtime + '/'
    parser.add_argument('--plotdir', type=str, default=plotdir)
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    checkpoints_dir = './results/checkpoints/' + strtime +  '/'
    parser.add_argument('--checkpoints_dir', type=str, default=checkpoints_dir)
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Generating the dictionary
    args = parser.parse_args()

    return args

# Probabilistic generalization of MAE
def crps_empirical(pred, truth):
    """
    Code borrowed from pyro: https://docs.pyro.ai/en/stable/_modules/pyro/ops/stats.html#crps_empirical

    Computes negative Continuous Ranked Probability Score CRPS* [1] between a
    set of samples ``pred`` and true data ``truth``. This uses an ``n log(n)``
    time algorithm to compute a quantity equal that would naively have
    complexity quadratic in the number of samples ``n``::

        CRPS* = E|pred - truth| - 1/2 E|pred - pred'|
              = (pred - truth).abs().mean(0)
              - (pred - pred.unsqueeze(1)).abs().mean([0, 1]) / 2

    Note that for a single sample this reduces to absolute error.

    **References**

    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param torch.Tensor pred: A set of sample predictions batched on rightmost dim.
        This should have shape ``(num_samples,) + truth.shape``.
    :param torch.Tensor truth: A tensor of true observations.
    :return: A tensor of shape ``truth.shape``.
    :rtype: torch.Tensor
    """
    if pred.shape[1:] != (1,) * (pred.dim() - truth.dim() - 1) + truth.shape:
        raise ValueError(
            "Expected pred to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
        )
    opts = dict(device=pred.device, dtype=pred.dtype)
    num_samples = pred.size(0)
    if num_samples == 1:
        return (pred[0] - truth).abs()

    pred = pred.sort(dim=0).values
    diff = pred[1:] - pred[:-1]
    weight = torch.arange(1, num_samples, **opts) * torch.arange(
        num_samples - 1, 0, -1, **opts
    )
    weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))

    return (pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples**2

class CRPSLoss(torch.nn.Module):
    def __init__(self):
        super(CRPSLoss, self).__init__()

    def forward(self, pred, truth):
        return crps_empirical(pred, truth)
    
class CI_MSELoss(torch.nn.Module):
    def __init__(self):
        super(CI_MSELoss, self).__init__()

    def forward(self, pred, truth):
        return torch.nn.functional.mse_loss(pred, truth) + torch.nn.functional.mse_loss(pred.mean(dim=(-2,-1)), truth.mean(dim=(-2,-1)))
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            torch.save(model.state_dict(), f"./last_best_model_hr.pt")
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(torch.load(f"./last_best_model_hr.pt"))
                return True, model
        return False, model

def train_step(model, dataloader, loss_fn, optimizer, epoch, device):

    """
    Function for training the UNet model for a single epoch.

    model: instance of the Unet class
    dataloader: torch training dataloader
    loss_fn: loss function
    optimizer: torch optimizer 
    epoch: current epoch
    device: device to use (GPU)

    return -> average loss value over the epoch
    """

    model.train()

    # Activating progress bar
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(f'Train :: Epoch: {epoch}')

        running_losses = {var: [] for var in ["pr", "tasmin", "tasmax"]}
        running_loss = []
        # Looping over the entire dataloader set
        for i, batch in enumerate(dataloader):
            tq.update(1)

            optimizer.zero_grad()

            # Extracting training data from batch
            inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            # Performing forward pass and computing loss
            preds = model(inputs, timestamps)
            loss = loss_fn(preds, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Logging per-variable training losses
            loss_pr = torch.nn.L1Loss()(preds[:,0,:,:], targets[:,0,:,:])
            loss_tasmin = torch.nn.L1Loss()(preds[:,1,:,:], targets[:,1,:,:])
            loss_tasmax = torch.nn.L1Loss()(preds[:,2,:,:], targets[:,2,:,:])

            running_losses['pr'] = running_losses['pr'] + [loss_pr.item()]
            running_losses['tasmin'] = running_losses['tasmin'] + [loss_tasmin.item()]
            running_losses['tasmax'] = running_losses['tasmax'] + [loss_tasmax.item()]

            tq.set_postfix_str(s=f'Loss: {(loss.item()):.4f}')
            running_loss.append(loss.item())

        mean_loss = sum(running_loss) / len(running_loss)
        tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')

        running_losses['pr'] = sum(running_losses['pr']) / len(running_losses['pr'])
        running_losses['tasmin'] = sum(running_losses['tasmin']) / len(running_losses['tasmin'])
        running_losses['tasmax'] = sum(running_losses['tasmax']) / len(running_losses['tasmax'])

        return running_losses

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
    batch = next(iter(dataloader))

    if dataloader.dataset.type == "lr_to_hr":
        inputs, lr, hr, timestamps, stand_stats = (batch['inputs'].to(device), batch['lr'], batch['hr'], batch['timestamps_float'], batch['stand_stats'])
        preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
        hr_pred = dataloader.dataset.invstand_residual(preds.detach().cpu(), stand_stats)
        fig, axs = dataloader.dataset.plot_batch(torch.nn.functional.interpolate(lr, size=hr.size()[-2:]), hr_pred, hr, timestamps, epoch)
    elif dataloader.dataset.type == "lrinterp_to_hr":
        inputs, lrinterp, hr, timestamps, stand_stats = (batch['inputs'].to(device), batch['lrinterp'], batch['hr'], batch['timestamps_float'], batch['stand_stats'])
        preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
        hr_pred = dataloader.dataset.invstand_residual(preds.detach().cpu(), stand_stats)
        fig, axs = dataloader.dataset.plot_batch(lrinterp, hr_pred, hr, timestamps, epoch)
    elif dataloader.dataset.type == "lrinterp_to_residuals" or dataloader.dataset.type == "lr_to_residuals":
        inputs, lrinterp, hr, timestamps, stand_stats = (batch['inputs'].to(device), batch['lrinterp'], batch['hr'], batch['timestamps_float'], batch['stand_stats'])
        preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
        hr_pred = dataloader.dataset.residual_to_hr(preds.detach().cpu(), lrinterp, stand_stats)
        fig, axs = dataloader.dataset.plot_batch(lrinterp, hr_pred, hr, timestamps, epoch)

    return hr_pred, (fig, axs)

@torch.no_grad()
def eval_model(model, dataloader, loss_fn, reconstruct, device, transfo=False):

    """
    Function for evaluating the unet model.

    model: instance of the Unet class
    dataloader: torch dataloader 
    loss_fn: metric used for evaluation
    reconstruct: if true the loss is computed on the high-resolution data
    device: device to use (GPU)
    transfo: if true, inverse transform the data before computing the loss

    return -> averaged loss over the dataloader set
    """

    model.eval()

    # Activating progress bar
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(':: Evaluation ::')

        running_losses = {var: [] for var in ["pr", "tasmin", "tasmax"]}
        # Looping over the entire dataloader set
        for i, batch in enumerate(dataloader):
            tq.update(1)

            # Extracting training data from batch and performing forward pass
            inputs, targets, timestamps, hr = (batch['inputs'].to(device), batch['targets'].to(device), batch['timestamps'], batch['hr'])
            preds = model(inputs, timestamps.unsqueeze(dim=1).to(device))

            # if we want to compute the loss on the high-resolution data
            if reconstruct:

                if dataloader.dataset.type == "lr_to_hr" or dataloader.dataset.type == "lrinterp_to_hr":
                    hr_preds = dataloader.dataset.invstand_residual(preds.detach().cpu(), batch['stand_stats'])
                elif dataloader.dataset.type == "lrinterp_to_residuals" or dataloader.dataset.type == "lr_to_residuals":
                    hr_preds = dataloader.dataset.residual_to_hr(preds.detach().cpu(), batch["lrinterp"], batch['stand_stats'])

                preds_hr = cu.kgm2sTommday(hr_preds[:, 0, :, :])
                preds_tasmin = cu.KToC(hr_preds[:, 1, :, :])
                preds_tasmax = cu.KToC(hr_preds[:, 2, :, :])
                if transfo:
                    preds_hr = cu.kgm2sTommday(cu.softplus_inv(preds[:, 0, :, :]))
                    preds_tasmin = cu.KToC(preds[:, 1, :, :])
                    preds_tasmax = cu.KToC(hr_preds[:, 2, :, :] + hr_preds[:, 1, :, :])

                hr_pr = cu.kgm2sTommday(hr[:, 0, :, :])
                hr_tasmin = cu.KToC(hr[:, 1, :, :])
                hr_tasmax = cu.KToC(hr[:, 2, :, :])
                if transfo:
                    hr_pr = cu.kgm2sTommday(cu.softplus_inv(hr[:, 0, :, :]))
                    hr_tasmin = cu.KToC(hr[:, 1, :, :])
                    hr_tasmax = cu.KToC(hr[:, 2, :, :] + hr[:, 1, :, :])

                loss_pr = torch.nn.L1Loss()(preds_hr, hr_pr)
                loss_tasmin = torch.nn.L1Loss()(preds_tasmin, hr_tasmin)
                loss_tasmax = torch.nn.L1Loss()(preds_tasmax, hr_tasmax)

                running_losses['pr'] = running_losses['pr'] + [loss_pr.item()]
                running_losses['tasmin'] = running_losses['tasmin'] + [loss_tasmin.item()]
                running_losses['tasmax'] = running_losses['tasmax'] + [loss_tasmax.item()]
            
            # if we want to compute the loss directly on the residual
            else:

                loss_pr = torch.nn.L1Loss()(preds[:,0,:,:], targets[:,0,:,:])
                loss_tasmin = torch.nn.L1Loss()(preds[:,1,:,:], targets[:,1,:,:])
                loss_tasmax = torch.nn.L1Loss()(preds[:,2,:,:], targets[:,2,:,:])

                running_losses['pr'] = running_losses['pr'] + [loss_pr.item()]
                running_losses['tasmin'] = running_losses['tasmin'] + [loss_tasmin.item()]
                running_losses['tasmax'] = running_losses['tasmax'] + [loss_tasmax.item()]

        running_losses['pr'] = sum(running_losses['pr']) / len(running_losses['pr'])
        running_losses['tasmin'] = sum(running_losses['tasmin']) / len(running_losses['tasmin'])
        running_losses['tasmax'] = sum(running_losses['tasmax']) / len(running_losses['tasmax'])

        return running_losses
