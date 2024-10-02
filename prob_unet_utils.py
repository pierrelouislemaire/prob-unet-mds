import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def truncated_normal_(tensor, mean=0.0, std=1.0):
    """
    Fills the input tensor with values drawn from a truncated normal distribution.
    
    Parameters:
    tensor (torch.Tensor): Tensor to be filled with truncated normal values.
    mean (float): Mean of the normal distribution. Default is 0.
    std (float): Standard deviation of the normal distribution. Default is 1.
    """
    # Generate random values from a normal distribution
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    
    # Apply truncation
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(dim=-1, keepdim=True)[1]
    
    # Copy the truncated values to the original tensor
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    """
    Initializes the weights of Conv2d and ConvTranspose2d layers using Kaiming Normal initialization.
    The biases are initialized using a truncated normal distribution.
    
    Parameters:
    m (torch.nn.Module): A layer of the neural network.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Apply Kaiming Normal initialization to weights
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
        # Apply truncated normal initialization to biases
        if m.bias is not None:
            truncated_normal_(m.bias, mean=0.0, std=0.001)

def init_weights_orthogonal_normal(m):
    """
    Initializes the weights of Conv2d and ConvTranspose2d layers using Orthogonal initialization.
    The biases are initialized using a truncated normal distribution.
    
    Parameters:
    m (torch.nn.Module): A layer of the neural network.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Apply Orthogonal initialization to weights
        nn.init.orthogonal_(m.weight)
        
        # Apply truncated normal initialization to biases
        if m.bias is not None:
            truncated_normal_(m.bias, mean=0.0, std=0.001)

def l2_regularization(model):
    """
    Computes the L2 regularization term for the given model.
    
    Parameters:
    model (torch.nn.Module): The neural network model for which L2 regularization is to be computed.
    
    Returns:
    torch.Tensor: The L2 regularization term.
    """
    l2_reg = torch.tensor(0.0, device=model.parameters().__next__().device)
    
    for param in model.parameters():
        l2_reg += param.norm(2)
    
    return l2_reg

def save_mask_prediction_example(mask, pred, iteration):
    """
    Saves the predicted mask and ground truth mask as images.
    
    Parameters:
    mask (torch.Tensor): The ground truth mask tensor.
    pred (torch.Tensor): The predicted mask tensor.
    iteration (int): The iteration number to be used in the file name.
    """
    plt.imshow(pred[0, :, :].cpu().numpy(), cmap='Greys')
    plt.savefig(f'images/{iteration}_prediction.png')
    
    plt.imshow(mask[0, :, :].cpu().numpy(), cmap='Greys')
    plt.savefig(f'images/{iteration}_mask.png')
