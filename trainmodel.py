import os
import warnings
import argparse
import torch

from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str, default='/home/julie/Data/Climex/day/kdj/')
    parser.add_argument('--variables', type=list, default=['pr', 'tasmin', 'tasmax'])
    parser.add_argument('--years_train', type=range, default=range(1960, 2059))
    parser.add_argument('--years_val', type=range, default=range(2060, 2079))
    parser.add_argument('--years_test', type=range, default=range(2080, 2098))
    parser.add_argument('--coords', type=list, default=[120, 184, 120, 184])
    parser.add_argument('--resolution', type=tuple, default=(64, 64))
    parser.add_argument('--lowres_scale', type=int, default=4)
    parser.add_argument('--timetransform', type=str, default='id', choices=['id', 'cyclic'])

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--optimizer', type=object, default=torch.optim.AdamW)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    strtime = datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    plotdir = './' + strtime + '/'
    parser.add_argument('--plotdir', type=str, default=plotdir)
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    args, unknown = parser.parse_known_args()

    return args

def train_step(model, dataloader, loss_fn, optimizer, scaler, step, accum, device):

    """
    Function for a single training step.
    :param model: instance of the Unet class
    :param loss_fn: loss function
    :param data_loader: data loader
    :param optimiser: optimiser to use
    :param scaler: scaler for mixed precision training
    :param step: current step
    :param accum: number of steps to accumulate gradients over
    :param writer: tensorboard writer
    :param device: device to use
    :return: loss value
    """

    model.train()

    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(f'Train :: Epoch: {step}')

        running_losses = []
        for i, batch in enumerate(dataloader):
            tq.update(1)

            inputs, targets = (batch['inputs'].to(device), batch['targets'].to(device))
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            with torch.cuda.amp.autocast():
                preds = model(inputs, class_labels=timestamps)
                loss = loss_fn(preds, targets)

            scaler.scale(loss).backward()

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
    Function for sampling the model.
    :param model: instance of the Unet class
    :param dataloader: data loader
    """

    model.eval()

    batch = next(iter(dataloader))

    inputs, lrinterp, hr, timestamps = (batch['inputs'].to(device), batch['lrinterp'], batch['hr'], batch['timestamps'])
    residual_preds = model(inputs, class_labels=timestamps.unsqueeze(dim=1).to(device))

    hr_pred = dataloader.dataset.residual_to_hr(residual_preds.detach().cpu(), lrinterp)

    fig, axs = dataloader.dataset.plot_batch(lrinterp, hr_pred, hr, timestamps, epoch)

    return hr_pred, (fig, axs)

@torch.no_grad()
def eval_model(model, dataloader, loss_fn, device):

    model.eval()

    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(':: Evaluation ::')

        running_losses = []
        for i, batch in enumerate(dataloader):
            tq.update(1)

            inputs, targets, timestamps = (batch['inputs'].to(device), batch['targets'].to(device), batch['timestamps'])
            residual_preds = model(inputs, class_labels=timestamps.unsqueeze(dim=1).to(device))

            loss = loss_fn(residual_preds, targets)
            running_losses.append(loss.item())

        mean_loss = sum(running_losses) / len(running_losses)
        tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')

    return mean_loss
