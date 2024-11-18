import warnings
import argparse
from tqdm import tqdm

import torch
import numpy as np

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

    # Downscaling method 
    parser.add_argument('--ds_model', type=str, default='deterministic_unet', choices=['deterministic_unet', 'probabilistic_unet', 'vae', 'linearcnn', 'bcsd'])

    # ML training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--optimizer', type=object, default=torch.optim.AdamW)

    # Model evaluation arguments
    #parser.add_argument('--metric', type=str, default="rmse", choices=["rmse", "crps", "rapsd", "mae"])

    # WandB activation 
    parser.add_argument('--wandb', type=bool, default=False)

    # GPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    """

    args, _ = parser.parse_known_args()
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

    """

    # Generating the dictionary
    args, _ = parser.parse_known_args()

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

    # Extracting batch data, forward pass, reconstruction (transformation, denormalization, etc.) and plotting
    # Procedure depends on the data pipeline 
    if dataloader.dataset.type == "lr_to_hr":
        inputs, lr, hr, timestamps = (batch['inputs'].to(device), batch['lr'], batch['hr'], batch['timestamps_float'])
        preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
        hr_pred = dataloader.dataset.invstand_residual(preds.detach().cpu())
        fig, axs = dataloader.dataset.plot_batch(torch.nn.functional.interpolate(lr, size=hr.size()[-2:]), hr_pred, hr, timestamps, epoch)
    elif dataloader.dataset.type == "lrinterp_to_hr":
        inputs, lrinterp, hr, timestamps = (batch['inputs'].to(device), batch['lrinterp'], batch['hr'], batch['timestamps_float'])
        preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
        hr_pred = dataloader.dataset.invstand_residual(preds.detach().cpu())
        fig, axs = dataloader.dataset.plot_batch(lrinterp, hr_pred, hr, timestamps, epoch)
    elif dataloader.dataset.type == "lrinterp_to_residuals" or dataloader.dataset.type == "lr_to_residuals":
        inputs, lrinterp, hr, timestamps = (batch['inputs'].to(device), batch['lrinterp'], batch['hr'], batch['timestamps_float'])
        preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
        hr_pred = dataloader.dataset.residual_to_hr(preds.detach().cpu(), lrinterp)
        fig, axs = dataloader.dataset.plot_batch(lrinterp, hr_pred, hr, timestamps, epoch)

    return hr_pred, (fig, axs)

@torch.no_grad()
def eval_model(model, dataloader, reconstruct, device, transfo=False):

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

        temporal = {var: [] for var in ["pr", "tasmin", "tasmax"]}
        spatial = {var: [] for var in ["pr", "tasmin", "tasmax"]}
        violations_count = {var: [] for var in ["pr", "temp"]}
        violations_avg = {var: [] for var in ["pr", "temp"]}
        running_losses = {var: [] for var in ["pr", "tasmin", "tasmax"]}

        # Looping over the entire dataloader set
        for i, batch in enumerate(dataloader):
            tq.update(1)

            # Extracting training data from batch and performing forward pass
            inputs, targets, timestamps, hr = (batch['inputs'].to(device), batch['targets'].to(device), batch['timestamps'], batch['hr'])
            preds = model(inputs, timestamps.unsqueeze(dim=1).to(device))

            # if we want to compute the loss on the high-resolution data
            if reconstruct:

                # reconstruct the high-resolution data
                if dataloader.dataset.type == "lr_to_hr" or dataloader.dataset.type == "lrinterp_to_hr":
                    hr_preds = dataloader.dataset.invstand_residual(preds.detach().cpu())
                elif dataloader.dataset.type == "lrinterp_to_residuals" or dataloader.dataset.type == "lr_to_residuals":
                    hr_preds = dataloader.dataset.residual_to_hr(preds.detach().cpu(), batch["lrinterp"])

                # inverse transform the data
                if transfo:
                    hr_preds[:, 0, :, :] = cu.softplus(hr_preds[:, 0, :, :])
                    hr_preds[:, 2, :, :] = cu.softplus(hr_preds[:, 2, :, :], c=0.) + hr_preds[:, 1, :, :]
                    hr[:, 0, :, :] = cu.softplus(hr[:, 0, :, :])
                    hr[:, 2, :, :] = cu.softplus(hr[:, 2, :, :], c=0.) + hr[:, 1, :, :]

                # to the original units
                preds_hr = cu.kgm2sTommday(hr_preds[:, 0, :, :])
                preds_tasmin = cu.KToC(hr_preds[:, 1, :, :])
                preds_tasmax = cu.KToC(hr_preds[:, 2, :, :])

                hr_pr = cu.kgm2sTommday(hr[:, 0, :, :])
                hr_tasmin = cu.KToC(hr[:, 1, :, :])
                hr_tasmax = cu.KToC(hr[:, 2, :, :])

                # log the mae per timestamp
                temporal["pr"].append(list(torch.abs(preds_hr - hr_pr).mean(dim=(-2,-1))))
                temporal["tasmin"].append(list(torch.abs(preds_tasmin - hr_tasmin).mean(dim=(-2,-1))))
                temporal["tasmax"].append(list(torch.abs(preds_tasmax - hr_tasmax).mean(dim=(-2,-1))))

                # log the mae per spatial point
                spatial["pr"].append(torch.abs(preds_hr - hr_pr).mean(dim=(0)))
                spatial["tasmin"].append(torch.abs(preds_tasmin - hr_tasmin).mean(dim=(0)))
                spatial["tasmax"].append(torch.abs(preds_tasmax - hr_tasmax).mean(dim=(0)))

                # log the number of constraint violations
                violations_count["pr"].append(torch.sum(preds_hr < 0, dim=0))
                violations_count["temp"].append(torch.sum(preds_tasmax < preds_tasmin, dim=0))

                # log the average constraint violation
                preds_hr[preds_hr > 0] = 0
                preds_tasmax[preds_tasmax > preds_tasmin] = preds_tasmin[preds_tasmax > preds_tasmin]
                constraint_error_pr = torch.sum(torch.abs(preds_hr), dim=0)
                constraint_error_temp = torch.sum(torch.abs(preds_tasmax - preds_tasmin), dim=0)
                violations_avg["pr"].append(constraint_error_pr)
                violations_avg["temp"].append(constraint_error_temp)
            
            # if we want to compute the loss directly on the residual
            else:

                loss_pr = torch.nn.L1Loss()(preds[:,0,:,:], targets[:,0,:,:])
                loss_tasmin = torch.nn.L1Loss()(preds[:,1,:,:], targets[:,1,:,:])
                loss_tasmax = torch.nn.L1Loss()(preds[:,2,:,:], targets[:,2,:,:])

                running_losses['pr'] = running_losses['pr'] + [loss_pr.item()]
                running_losses['tasmin'] = running_losses['tasmin'] + [loss_tasmin.item()]
                running_losses['tasmax'] = running_losses['tasmax'] + [loss_tasmax.item()]

        if reconstruct:

            # joining the lists of mae per timestamp
            temporal["pr"] = sum(temporal["pr"], [])
            temporal["tasmin"] = sum(temporal["tasmin"], [])
            temporal["tasmax"] = sum(temporal["tasmax"], [])

            # averaging the mae per spatial point over the batches
            spatial["pr"] = (np.sum(spatial["pr"], axis=0) / len(spatial["pr"])).flatten()
            spatial["tasmin"] = (np.sum(spatial["tasmin"], axis=0) / len(spatial["tasmin"])).flatten()
            spatial["tasmax"] = (np.sum(spatial["tasmax"], axis=0) / len(spatial["tasmax"])).flatten()

            # summing the number of constraint violations over the batches
            violations_count["pr"] = np.sum(violations_count["pr"], axis=0).flatten()
            violations_count["temp"] = np.sum(violations_count["temp"], axis=0).flatten()

            # if the number of constraint violations is 0, we set it to 1 to avoid division by 0
            tmp_violcount_pr = violations_count['pr']
            tmp_violcount_temp = violations_count['temp']
            tmp_violcount_pr[tmp_violcount_pr == 0] = 1
            tmp_violcount_temp[tmp_violcount_temp == 0] = 1

            # averaging the constraint violations over the batches
            violations_avg["pr"] = np.sum(violations_avg["pr"], axis=0).flatten() / tmp_violcount_pr
            violations_avg["temp"] = np.sum(violations_avg["temp"], axis=0).flatten() / tmp_violcount_temp

            return temporal, spatial, violations_count, violations_avg
        
        else: 
            running_losses['pr'] = sum(running_losses['pr']) / len(running_losses['pr'])
            running_losses['tasmin'] = sum(running_losses['tasmin']) / len(running_losses['tasmin'])
            running_losses['tasmax'] = sum(running_losses['tasmax']) / len(running_losses['tasmax'])

            return running_losses
