import torch
import matplotlib.pyplot as plt
# from dask.distributed import Client

import climex_utils as cu
import train_prob_unet_model as tm  
from prob_unet import ProbabilisticUNet
from prob_unet_utils import plot_losses
from accelerate import Accelerator
import pickle
  

if __name__ == "__main__":

    # # Initialize Dask Client once
    # client = Client()
    # Importing all required arguments
    args = tm.get_args()

    # Initializing the Probabilistic UNet model
    probunet_model = ProbabilisticUNet(
        input_channels=len(args.variables),
        num_classes=len(args.variables),
        latent_dim=6,
        num_filters=[64, 128, 256, 512],
        beta=args.beta
    ).to(args.device)

    # Initializing the datasets
    dataset_train = cu.climex2torch(
        datadir=args.datadir,
        years=args.years_train,
        variables=args.variables,
        type="lrinterp_to_residuals",
        transfo=True,
        coords=args.coords,
        lowres_scale=args.lowres_scale,

    )
    
    dataset_val = cu.climex2torch(
        datadir=args.datadir,
        years=args.years_val,
        variables=args.variables,
        coords=args.coords,
        lowres_scale=args.lowres_scale,
        type="lrinterp_to_residuals",
        transfo=True
    )
    dataset_test = cu.climex2torch(
        datadir=args.datadir,
        years=args.years_test,
        variables=args.variables,
        coords=args.coords,
        lowres_scale=args.lowres_scale,
        type="lrinterp_to_residuals",
        transfo=True
    )

    # Initializing the dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    dataloader_test_random = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    # Initializing training objects
    optimizer = args.optimizer(params=probunet_model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(probunet_model.parameters(), lr=args.lr, weight_decay=1e-4)


    # Initialize loss tracking lists for each variable
    tr_losses_mae = {var: [] for var in args.variables}
    tr_losses_kl = {var: [] for var in args.variables}
    val_losses_mae = {var: [] for var in args.variables}
    val_losses_kl = {var: [] for var in args.variables}

    # Initialize loss storage dictionaries
    all_train_losses_mae = {var: [] for var in args.variables}
    all_train_losses_kl = {var: [] for var in args.variables}
    all_val_losses_mae = {var: [] for var in args.variables}
    all_val_losses_kl = {var: [] for var in args.variables}

    # initial_beta = 0
    # max_beta = args.beta
    # num_warmup_epochs = 5
    # args.num_epochs = 30
    

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        
        # if epoch <= num_warmup_epochs:
        #     # Keep beta at zero for the warmup period
        #     current_beta = 0
        # else:
        #     # Gradually increase beta from 0 to max_beta over the remaining epochs
        #     progress = (epoch - num_warmup_epochs) / (args.num_epochs - num_warmup_epochs)
        #     current_beta = progress * max_beta
        # # Gradually increase beta
        # # current_beta = min(initial_beta + epoch * (max_beta / args.num_epochs), max_beta)

        # # Ensure current_beta does not exceed max_beta
        # current_beta = min(current_beta, max_beta)
        
        # # Set the current beta for the model
        # probunet_model.beta = current_beta

        # Training for one epoch
        train_losses_mae, training_losses_kl = tm.train_probunet_step(
            model=probunet_model,
            dataloader=dataloader_train,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=args.num_epochs,
            device=args.device,
            variables=args.variables,
        )
        for var in args.variables:
            tr_losses_mae[var].append(train_losses_mae[var])
            tr_losses_kl[var].append(training_losses_kl[var])
            # Store losses in epoch dictionaries
            all_train_losses_mae[var].append(train_losses_mae[var])
            all_train_losses_kl[var].append(training_losses_kl[var])

        # Evaluating the model on validation data
        val_losses_mae_running, val_losses_kl_running = tm.eval_probunet_model(
            model=probunet_model,
            dataloader=dataloader_val,
            reconstruct=False,
            device=args.device,           
        )
        for var in args.variables:
            val_losses_mae[var].append(val_losses_mae_running[var])
            val_losses_kl[var].append(val_losses_kl_running[var])

            # Store losses in epoch dictionaries
            all_val_losses_mae[var].append(val_losses_mae_running[var])
            all_val_losses_kl[var].append(val_losses_kl_running[var])


        # Sampling from the model every 2 epochs
        # if epoch % 2 == 0:
        samples, (fig, axs) = tm.sample_probunet_model(
            model=probunet_model,
            dataloader=dataloader_test_random,
            epoch=epoch,
            device=args.device
        )
        # Save sample plots
        fig.savefig(f"{args.plotdir}/epoch{epoch}.png", dpi=300)
        plt.close(fig)
    
        # Save losses to a file after training
    losses_to_save = {
        "train_losses_mae": all_train_losses_mae,
        "train_losses_kl": all_train_losses_kl,
        "val_losses_mae": all_val_losses_mae,
        "val_losses_kl": all_val_losses_kl,
    }
    with open(f"{args.plotdir}/losses.pkl", "wb") as f:
        pickle.dump(losses_to_save, f)


    # Plot training and validation loss curves for each variable
    plot_losses(tr_losses_mae, tr_losses_kl, val_losses_mae, val_losses_kl, args.variables, args.plotdir)

    # # Close the Dask Client after training is complete
    # client.close()