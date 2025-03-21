import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Independent, kl
from codebase.models.deterministic_unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class ResidualBlock(nn.Module):
    
    """
    Residual block used in the prior and posterior networks.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            x (torch.Tensor): Output tensor.
        """
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.res_conv(residual)
        x = self.relu(x)

        return x

class AxisAlignedConvGaussian(nn.Module):

    """
    Axis-Aligned Convolutional Gaussian distribution for the latent space.
    This module computes the mean (mu) and log of standard deviation (log_sigma)
    of the Gaussian distribution using convolutional layers.
    """

    def __init__(self, input_channels, num_filters, latent_dim, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.posterior = posterior

        # If posterior, the input will include the target concatenated
        if posterior:
            self.input_channels += input_channels  # Concatenate input and target

        # Define the encoder
        layers = []
        in_channels = self.input_channels

        # Build the encoder using convolutional layers
        for out_channels in num_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # Convolutional layers to compute mu and log_sigma
        self.conv_mu = nn.Conv2d(num_filters[-1], latent_dim, kernel_size=1)
        self.conv_log_sigma = nn.Conv2d(num_filters[-1], latent_dim, kernel_size=1)

        self.apply(init_weights)

    def forward(self, x, target=None):

        """
        Forward pass to compute the distribution of the latent variable.

        Args:
            x (torch.Tensor): Input tensor.
            target (torch.Tensor, optional): Target tensor (for posterior).

        Returns:
            dist (torch.distributions.Distribution): The computed Gaussian distribution.
        """
        # Concatenate input and target for posterior
        if self.posterior and target is not None:
            x = torch.cat([x, target], dim=1)

        # Encode the input to get the latent features
        h = self.encoder(x)
        # print("-------------------------------------------")
        # print("After encoder:", torch.isnan(h).any(), h.min(), h.max())

        # Global average pooling to get a single vector per sample
        h = torch.mean(h, dim=[2, 3], keepdim=True)
        # print("-------------------------------------------")
        # print("After mean pooling:", torch.isnan(h).any(), h.min(), h.max())

        # Compute mu and log_sigma
        mu = self.conv_mu(h)
        log_sigma = self.conv_log_sigma(h)
        # print("-------------------------------------------")
        # print("mu:", torch.isnan(mu).any(), mu.min(), mu.max())
        # print("-------------------------------------------")
        # print("log_sigma:", torch.isnan(log_sigma).any(), log_sigma.min(), log_sigma.max())

        # Remove the extra dimensions (height and width dimensions are 1 after pooling)
        mu = mu.squeeze(-1).squeeze(-1)
        log_sigma = log_sigma.squeeze(-1).squeeze(-1)

        # This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        # Create a Normal distribution with the computed parameters
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma) + 1e-7), 1)
        return dist

class Fcomb(nn.Module):

    """
    Combines the UNet features with the latent variable z to produce the final output.
    """

    def __init__(self, unet_output_channels, latent_dim, num_classes):
        super(Fcomb, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]

        # Define the layers to combine UNet features and latent variable
        self.layers = nn.Sequential(
            nn.Conv2d(unet_output_channels + latent_dim, unet_output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_output_channels, unet_output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_output_channels, num_classes, kernel_size=1)
        )

        self.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function mimics TensorFlow's `tile()` function for PyTorch.
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*repeat_idx)
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):

        """
        Forward pass to combine UNet features with latent variable.

        Args:
            feature_map (torch.Tensor): Feature map from UNet.
            z (torch.Tensor): Sampled latent variable.

        Returns:
            output (torch.Tensor): The final output tensor.
        """
        # Expand z to match the spatial dimensions of the feature map
        # Tile z to match feature map size
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

        # Concatenate feature map and latent variable
        h = torch.cat([feature_map, z], dim=1)

        # Pass through the combination layers
        output = self.layers(h)
        return output

class LatentAmplificator(nn.Module):
    def __init__(self, latent_dim, feat_dim, resolution):
        super(LatentAmplificator, self).__init__()

        self.resolution_flat = resolution[0] * resolution[1]
        self.resolution = resolution

        self.fc = nn.Linear(latent_dim, self.resolution_flat)
        
        self.conv1 = nn.Conv2d(1, feat_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feat_dim // 2, feat_dim, kernel_size=3, padding=1)

        self.act = nn.SiLU()
        self.bn2 = nn.BatchNorm2d(feat_dim // 2)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 1, self.resolution[0], self.resolution[1])
        z = self.conv1(self.act(z))
        z = self.conv2(self.bn2(self.act(z)))
        return z
        

class ProbabilisticUNet(nn.Module):

    """
    The Probabilistic U-Net model combining a U-Net backbone with a variational latent space.
    """

    def __init__(self, input_channels, num_classes, latent_dim=6, num_filters=[64, 128, 256, 512], beta_0 = 1.0, beta_1=1.0, beta_2=1.0, use_geco=False):
        super(ProbabilisticUNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.proj_latent_dim = 64
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.use_geco = use_geco
        self.lagrange_mult = 1.0
        self.threshold_k = 0.16
        self.mae_alpha = 0.9
        self.constraint_ma = 0
        self.constraint = 0

        # Initialize the U-Net backbone
        self.unet = UNet(
            img_resolution=(128, 128),  
            in_channels=input_channels,
            model_channels=64,
            out_channels=num_filters[0],
            label_dim=1,
            use_diffuse=False
        ).to(device)

        # Prior network (without target)
        self.prior = AxisAlignedConvGaussian(
            input_channels=input_channels,
            num_filters=num_filters,
            latent_dim=latent_dim,
            posterior=False
        ).to(device)

        # Posterior network (with target)
        self.posterior = AxisAlignedConvGaussian(
            input_channels=input_channels,
            num_filters=num_filters,
            latent_dim=latent_dim,
            posterior=True
        ).to(device)

        #self.latent_amplificator = LatentAmplificator(latent_dim, self.proj_latent_dim, (128, 128))

        # Combines UNet features and latent variable to produce the output
        self.fcomb = Fcomb(
            unet_output_channels=num_filters[0],
            latent_dim=self.latent_dim,
            num_classes=num_classes
        ).to(device)

        # Apply Kaiming initialization to all the convolutional layers
        # self.apply(init_weights)  # It has already been applied to the individual components

    def forward(self, x, target=None, t=None, training=True):

        """
        Forward pass of the Probabilistic U-Net.

        Args:
            x (torch.Tensor): Input tensor.
            target (torch.Tensor, optional): Target tensor (for training).
            training (bool): Flag indicating whether in training mode.

        Returns:
            output (torch.Tensor): The model's output tensor.
        """

        # Get features from the UNet backbone      
        unet_features = self.unet(x, t)

        # During training, sample z from the posterior
        if training and target is not None:
            self.posterior_latent_space = self.posterior(x, target)
            z = self.posterior_latent_space.rsample()
    

        # During inference, sample z from the prior
        else:
            self.prior_latent_space = self.prior(x)
            z = self.prior_latent_space.rsample()
        
        #z = self.latent_amplificator(z)
        output = self.fcomb(unet_features, z)
        return output
    
    def elbo(self, x, target, t):

        """
        Computes the Evidence Lower Bound (ELBO) loss for training.

        Args:
            x (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            total_loss (torch.Tensor): The total ELBO loss.
            recon_loss (torch.Tensor): The reconstruction loss component.
            kl_div (torch.Tensor): The KL divergence component.
        """

         # Get features from the UNet backbone      
        unet_features = self.unet(x, t)

        # Compute prior and posterior distributions
        self.prior_latent_space = self.prior(x)
        self.posterior_latent_space = self.posterior(x, target)

        # Sample z from the posterior
        z_posterior = self.posterior_latent_space.rsample()

        #z_posterior = self.latent_amplificator(z_posterior)

        # Compute the output
        output = self.fcomb(unet_features, z_posterior)

        # Initialize total reconstruction loss and list for individual variable losses
        total_recon_loss = 0
        recon_loss_list = []

        for i in range(output.shape[1]):  
            # Compute reconstruction loss for each variable
            recon_loss = nn.L1Loss(reduction='mean')(output[:, i, :, :], target[:, i, :, :])
            recon_loss_list.append(recon_loss.item())  # Store individual variable loss
        
        total_recon_loss = nn.L1Loss()(output, target)  # Average to total loss

        # KL divergence between posterior and prior
        kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        kl_div = torch.mean(kl_div)

        # Define the standard Gaussian distribution
        standard_gaussian = Independent(
            Normal(
                loc=torch.zeros_like(self.posterior_latent_space.base_dist.loc).to(device),
                scale=torch.ones_like(self.posterior_latent_space.base_dist.scale).to(device)
            ),
            1
        )

        # KL divergence between posterior and standard Gaussian
        kl_div2 = kl.kl_divergence(self.posterior_latent_space, standard_gaussian)
        kl_div2 = torch.mean(kl_div2)

        total_loss = self.beta_0 * total_recon_loss + self.beta_1 * kl_div + self.beta_2 * kl_div2

        return total_loss, total_recon_loss, kl_div, kl_div2
    
    def geco(self, x, target, t):

        # Get features from the UNet backbone      
        unet_features = self.unet(x, t)

        # Compute prior and posterior distributions
        self.prior_latent_space = self.prior(x)
        self.posterior_latent_space = self.posterior(x, target)

        # Sample z from the posterior
        z_posterior = self.posterior_latent_space.rsample()

        #z_posterior = self.latent_amplificator(z_posterior)

        # Compute the output
        output = self.fcomb(unet_features, z_posterior)

        self.constraint = torch.nn.L1Loss()(output, target) - self.threshold_k
        
        kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        kl_div = torch.mean(kl_div)

        geco_loss = kl_div + self.lagrange_mult * self.constraint
        recon_loss = self.constraint + self.threshold_k

        return geco_loss, recon_loss, kl_div