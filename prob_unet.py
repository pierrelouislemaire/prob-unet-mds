import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, kl
from networks import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # Convolutional layers to compute mu and log_sigma
        self.conv_mu = nn.Conv2d(num_filters[-1], latent_dim, kernel_size=1)
        self.conv_log_sigma = nn.Conv2d(num_filters[-1], latent_dim, kernel_size=1)

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

        # Global average pooling to get a single vector per sample
        h = torch.mean(h, dim=[2, 3], keepdim=True)

        # Compute mu and log_sigma
        mu = self.conv_mu(h)
        log_sigma = self.conv_log_sigma(h)

        # Remove the extra dimensions (height and width dimensions are 1 after pooling)
        mu = mu.squeeze(-1).squeeze(-1)
        log_sigma = log_sigma.squeeze(-1).squeeze(-1)

        # This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        # Create a Normal distribution with the computed parameters
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist

class Fcomb(nn.Module):

    """
    Combines the UNet features with the latent variable z to produce the final output.
    """

    def __init__(self, unet_output_channels, latent_dim, num_classes):
        super(Fcomb, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Define the layers to combine UNet features and latent variable
        self.layers = nn.Sequential(
            nn.Conv2d(unet_output_channels + latent_dim, unet_output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_output_channels, unet_output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_output_channels, num_classes, kernel_size=1)
        )

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
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = z.expand(-1, -1, feature_map.size(2), feature_map.size(3))

        # Concatenate feature map and latent variable
        h = torch.cat([feature_map, z], dim=1)

        # Pass through the combination layers
        output = self.layers(h)
        return output

class ProbabilisticUNet(nn.Module):

    """
    The Probabilistic U-Net model combining a U-Net backbone with a variational latent space.
    """

    def __init__(self, input_channels, num_classes, latent_dim=6, num_filters=[64, 128, 256, 512], beta=1.0):
        super(ProbabilisticUNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.beta = beta

        # Initialize the U-Net backbone
        self.unet = UNet(
            img_resolution=(64, 64),  
            in_channels=input_channels,
            out_channels=num_filters[0],
            label_dim=0,
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

        # Combines UNet features and latent variable to produce the output
        self.fcomb = Fcomb(
            unet_output_channels=num_filters[0],
            latent_dim=latent_dim,
            num_classes=num_classes
        ).to(device)

    def forward(self, x, target=None, training=True):

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
        unet_features = self.unet(x)

        # During training, sample z from the posterior
        if training and target is not None:
            self.posterior_latent_space = self.posterior(x, target)
            z = self.posterior_latent_space.rsample()

        # During inference, sample z from the prior
        else:
            self.prior_latent_space = self.prior(x)
            z = self.prior_latent_space.rsample()
        
        output = self.fcomb(unet_features, z)
        return output
    
    def elbo(self, x, target):

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
        unet_features = self.unet(x)

        # Compute prior and posterior distributions
        self.prior_latent_space = self.prior(x)
        self.posterior_latent_space = self.posterior(x, target)

        # Sample z from the posterior
        z_posterior = self.posterior_latent_space.rsample()

        # Compute the output
        output = self.fcomb(unet_features, z_posterior)

        # Reconstruction loss (Mean Squared Error)
        recon_loss = nn.MSELoss(reduction='sum')(output, target)

        # KL divergence between posterior and prior
        kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space).sum()

        total_loss = recon_loss + self.beta * kl_div

        return total_loss, recon_loss, kl_div