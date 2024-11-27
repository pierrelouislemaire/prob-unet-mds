import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

# For plotting the smoothed training and validation losses
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# def plot_losses(train_losses_mae, train_losses_kl, val_losses_mae, val_losses_kl, variables, plotdir):
#     """
#     Plots the training and validation losses for each variable and saves the figure.
#     """
#     for var in variables:
#         # train_loss = moving_average(np.array(train_losses[var]), w=24)
#         # val_loss = moving_average(np.array(val_losses[var]), w=48)
#         fig, ax = plt.subplots(figsize=(15,10))
#         ax.plot(train_losses_mae[var], lw=2, color='blue', label='training')
#         ax.plot(val_losses_mae[var], lw=2, linestyle='dashed', color="blue", label='validation')
#         ax.set_xlabel("Epochs")
#         ax.set_ylabel("MAE Loss", color='blue')
#         ax.tick_params(axis='y', labelcolor='blue')

#         ax1 = ax.twinx()
#         ax1.plot(train_losses_kl[var], lw=2, color='red', label='training KL')
#         ax1.plot(val_losses_kl[var], lw=2, linestyle='dashed', color='red', label='validation KL loss')
#         ax1.set_ylabel("KL Loss", color='red')
#         ax1.tick_params(axis='y', labelcolor='red')
#         plt.title(f"MAE & KL Losses for {var}")
#         plt.legend()
#         plt.savefig(f'{plotdir}/{var}_loss.png', dpi=300)
#         plt.close()

def plot_losses(train_losses_mae, train_losses_kl, val_losses_mae, val_losses_kl, variables, plotdir):
    """
    Plots the training and validation losses for each variable and saves the figure.
    """
    for var in variables:
        fig, ax = plt.subplots(figsize=(15,10))  # Use plt.subplots instead of plt.subplot

        # Ensure that the loss data is converted to NumPy arrays
        train_mae = np.array(train_losses_mae[var])
        val_mae = np.array(val_losses_mae[var])
        train_kl = np.array([loss.detach().cpu().item() for loss in train_losses_kl[var]])
        val_kl = np.array([loss.detach().cpu().item() for loss in val_losses_kl[var]])

        # Exclude the first epoch loss values to smooth the plot
        train_mae = train_mae[1:]
        val_mae = val_mae[1:]
        train_kl = train_kl[1:]
        val_kl = val_kl[1:]

        # Generate epoch numbers starting from 2
        epochs = np.arange(2, len(train_mae) + 2)  # +2 because we start from epoch 2

        # Plot MAE losses
        ax.plot(epochs, train_mae, lw=2, color='blue', label='Training MAE')
        ax.plot(epochs, val_mae, lw=2, linestyle='dashed', color='blue', label='Validation MAE')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('MAE Loss', color='blue', fontsize=14)
        ax.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for KL loss
        ax1 = ax.twinx()

        # Plot KL losses
        ax1.plot(epochs, train_kl, lw=2, color='red', label='Training KL')
        ax1.plot(epochs, val_kl, lw=2, linestyle='dashed', color='red', label='Validation KL')
        ax1.set_ylabel('KL Loss', color='red', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='red')

        # Combine legends from both axes
        lines_labels = [ax.get_legend_handles_labels(), ax1.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        ax.legend(lines, labels, loc='upper right')

        ax.set_title(f'Training and Validation Losses for {var}', fontsize=16)
        ax.grid(True)
        fig.tight_layout()

        plt.show()

        # Save the figure
        # fig.savefig(f"{plotdir}/loss_{var}.png", dpi=300)
        plt.close(fig)


