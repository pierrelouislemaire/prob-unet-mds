import warnings
import argparse
from tqdm import tqdm
import torch

warnings.filterwarnings('ignore')

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():

    """
    This function returns a dictionary containing all necessary arguments for importing ClimEx data, training, evaluating and sampling from a downscaling ML model.
    This function is helpful for doing sweeps and performing hyperparameter tuning.
    """

    parser = argparse.ArgumentParser()

    # climate dataset arguments
    parser.add_argument('--datadir', type=str, default='/home/julie/Data/Climex/day/kdj/')
    parser.add_argument('--variables', type=list, default=['pr', 'tas'])
    parser.add_argument('--years_train', type=range, default=range(1960, 1990))
    parser.add_argument('--years_subtrain', type=range, default=range(1960, 1980))
    parser.add_argument('--years_earlystop', type=range, default=range(1980, 1990))
    parser.add_argument('--years_val', type=range, default=range(1990, 1998))
    parser.add_argument('--years_megatrain', type=range, default=range(1960, 2000))
    parser.add_argument('--years_test', type=range, default=range(2000, 2010))
    parser.add_argument('--coords', type=list, default=[80, 208, 100, 228])
    parser.add_argument('--resolution', type=tuple, default=(128, 128))
    parser.add_argument('--lowres_scale', type=int, default=16)
    parser.add_argument('--transfo', type=bool, default=True)
    parser.add_argument('--pipeline', type=str, default='lrinterp_to_residuals', choices=['lrinterp_to_residuals', 'lr_to_residuals', 'lrinterp_to_hr', 'lr_to_hr'])

    # Model 
    parser.add_argument('--model', type=str, default='deterministic_unet', choices=['deterministic_unet', 'probabilistic_unet'])

    # ML training arguments
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=object, default=torch.optim.AdamW)

    # Probabilistic model arguments
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_filters', type=list, default=[64, 128, 256, 512])
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=32)

    # GPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

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
            torch.save(model.state_dict(), f"./last_best_model.pt")
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # if the counter is greater than the patience, load the best model and return True to break training
            if self.counter >= self.patience:
                model.load_state_dict(torch.load(f"./last_best_model.pt"))
                return True, model
        return False, model
    
def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def train_step(model, dataloader, loss_fn, optimizer, epoch, prob, device):

    """
    Function for training the UNet model (deterministic or probabilistic) for a single epoch.

    model: instance of the Unet class
    dataloader: torch training dataloader
    loss_fn: loss function
    optimizer: torch optimizer 
    epoch: current epoch
    prob: if True, the model is probabilistic
    device: device to use (GPU)

    return -> average loss values for each variable
    """

    model.train()

    # Activating progress bar
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(f'Train :: Epoch: {epoch}')

        variables = dataloader.dataset.variables

        running_losses_mae = []
        running_losses_kl = []
        running_losses_kl2 = []
        step_losses = []

        # Looping over the entire dataloader set
        for i, batch in enumerate(dataloader):
            tq.update(1)

            optimizer.zero_grad()

            # Extracting training data from batch
            inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            # Performing forward pass and computing loss
            if prob:
                if model.use_geco:
                    loss, recon_loss, kl_div = model.geco(inputs, targets, timestamps)
                else:
                    loss, recon_loss, kl_div, kl_div2 = model.elbo(inputs, targets, timestamps)
                reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.fcomb.layers)
                loss = loss + 1e-5 * reg_loss
            else:
                preds = model(inputs, timestamps)
                loss = loss_fn(preds, targets)
            
            # Backward pass
            loss.backward()

            if prob:
                if model.use_geco:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    with torch.no_grad():
                        if model.constraint_ma == 0:
                            model.constraint_ma = model.constraint.detach()
                        else:
                            model.constraint_ma = model.mae_alpha * model.constraint_ma.detach() + (1 - model.mae_alpha) * model.constraint
                        model.lagrange_mult *= torch.exp(0.1*model.constraint_ma)

            optimizer.step()

            # Log losses for each variable
            if prob:
                running_losses_mae.append(recon_loss.item())
                running_losses_kl.append(kl_div.detach().cpu())
                running_losses_kl2.append(kl_div2.detach().cpu())
            else:
                recon_loss = loss
                running_losses_mae.append(recon_loss.item())
                
            tq.set_postfix_str(s=f'Loss: {(recon_loss):.4f}')
            step_losses.append(recon_loss)

        mean_loss = sum(step_losses) / len(step_losses)
        tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')

        epoch_losses_mae = sum(running_losses_mae) / len(running_losses_mae)
        if prob:
            epoch_losses_kl = sum(running_losses_kl) / len(running_losses_kl)
            epoch_losses_kl2 = sum(running_losses_kl2) / len(running_losses_kl2)
            if model.use_geco:
                return epoch_losses_mae, epoch_losses_kl, model.lagrange_mult.item()
            else:
                return epoch_losses_mae, epoch_losses_kl, epoch_losses_kl2
        else:
            return epoch_losses_mae

@torch.no_grad()
def sample_model(model, dataloader, epoch, prob, num_samples, device):

    """
    Function for sampling from the unet model and plotting results.

    model: instance of the Unet class
    dataloader: torch dataloader (should be shuffled)
    epoch: last training epoch
    prob: if True, the model is probabilistic
    num_samples: number of samples for the probabilistic model
    device: device to use (GPU)

    return -> predicted high-resolution samples, plots
    """

    model.eval()
    batch = next(iter(dataloader))

    inputs, lrinterp, hr, timestamps, timestamps_float = (batch['inputs'].to(device), batch['lrinterp'], batch['hr'], batch["timestamps"], batch['timestamps_float'])
    if prob:
        preds = []
        for _ in range(num_samples):
            output = model(inputs, t=timestamps.unsqueeze(dim=1).to(device), training=False) # Generate output from the model
            pred = dataloader.dataset.invstand_residual(output.cpu())  # Convert residual to high-res
            preds.append(pred) # Append the prediction to the list
        preds = torch.stack(preds, dim=1)  # Shape: [batch_size, num_samples, channels, height, width]
    else:
        preds = model(inputs, timestamps.unsqueeze(dim=1).to(device))

    if dataloader.dataset.type == "lr_to_hr":
        hr_preds = dataloader.dataset.invstand_residual(preds.detach().cpu())
    elif dataloader.dataset.type == "lrinterp_to_hr":
        hr_preds = dataloader.dataset.invstand_residual(preds.detach().cpu())
    elif dataloader.dataset.type == "lrinterp_to_residuals" or dataloader.dataset.type == "lr_to_residuals":
        if prob:
            lrinterp = lrinterp.unsqueeze(1)
        hr_preds = dataloader.dataset.residual_to_hr(preds.detach().cpu(), lrinterp)

    fig, axs = dataloader.dataset.plot_batch(lrinterp.cpu(), hr_preds.cpu(), hr.cpu(), timestamps_float, epoch, N=2)

    return hr_preds, (fig, axs)

@torch.no_grad()
def eval_model(model, dataloader, prob, device):

    """
    Function for evaluating the unet model.

    model: instance of the Unet class
    dataloader: torch dataloader 
    loss_fn: metric used for evaluation
    prob: if True, the model is probabilistic
    device: device to use (GPU)

    return -> averaged loss over the dataloader set
    """

    model.eval()

    # Activating progress bar
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(':: Evaluation ::')

        num_samples_eval = 5
        variables = dataloader.dataset.variables
        eval_mae = {var: [] for var in variables}

        # Looping over the entire dataloader set
        for i, batch in enumerate(dataloader):
            tq.update(1)

            # Extracting training data from batch and performing forward pass
            inputs, targets, timestamps, hr = (batch['inputs'].to(device), batch['targets'].to(device), batch['timestamps'].unsqueeze(dim=1).to(device), batch['hr'])
            if prob:
                preds = []
                for s in range(num_samples_eval):
                    preds.append(model(inputs, t=timestamps, training=False))
                preds = torch.mean(torch.stack(preds, dim=1), dim=1)
            else:
                preds = model(inputs, timestamps.to(device))

            for i, var in enumerate(variables):
                eval_mae[var].append(torch.nn.L1Loss()(preds[:,i,:,:], targets[:,i,:,:]).item())

        for var in variables:
            eval_mae[var] = sum(eval_mae[var]) / len(eval_mae[var])

        return eval_mae
