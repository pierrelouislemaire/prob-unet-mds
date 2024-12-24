import torch
import matplotlib.pyplot as plt
# from dask.distributed import Client

import climex_utils as cu
import train_prob_unet_model as tm  
from prob_unet import ProbabilisticUNet
from prob_unet_utils import plot_losses, plot_losses_mae
from accelerate import Accelerator
import pickle
import numpy as np
  

if __name__ == "__main__":

    # Importing all required arguments
    args = tm.get_args()

    # # Increase the beta and beta_2 values simultaneously
    # # -----------------------------------------------------------------------------------
    # # Initialize scheduling parameters
    # max_beta = 0.8            
    # max_beta_2 = 0.8       
    # num_epochs = args.num_epochs = 40
    # start_beta = 0.0
    # start_beta_2 = 0.0       

    # # Calculate increment per epoch for beta and beta_2
    # if num_epochs > 0:
    #     beta_increment = (max_beta - start_beta) / num_epochs
    #     beta_2_increment = (max_beta_2 - start_beta_2) / num_epochs
    # else:
    #     beta_increment = 0.0
    #     beta_2_increment = 0.0
    # # -----------------------------------------------------------------------------------

    # # Increasing beta and beta_2 gradually when the loss contains two KL terms
    # # -----------------------------------------------------------------------------------
    # # Initialize beta and beta_2 schedules
    # max_beta_2 = 0.8
    # warmup_epochs = args.warmup_epochs = 0
    # beta_2_schedule_fraction = args.beta_2_schedule_fraction  = 1
    # args.num_epochs = 50
    # args.lowres_scale = 8

    # # Calculate beta_2 targets and increments
    # beta_2_target_first_phase = beta_2_schedule_fraction * max_beta_2  
    # remaining_beta_2 = max_beta_2 - beta_2_target_first_phase  
    # second_phase_epochs = args.num_epochs - warmup_epochs
    # if second_phase_epochs > 0:
    #     beta_2_increment_second_phase = remaining_beta_2 / second_phase_epochs
    # else:
    #     beta_2_increment_second_phase = 0.0
    
    # # Calculate beta increments after warmup
    # max_beta = 0.8
    # beta_increment_second_phase = max_beta / second_phase_epochs if second_phase_epochs > 0 else 0.0

    # beta_2_increment_first_phase = beta_2_target_first_phase / (warmup_epochs + 1e-6)  # Add a small value to avoid division by zero
    # # ------------------------------------------------------------------------------------

    # Initializing the Probabilistic UNet model
    probunet_model = ProbabilisticUNet(
        input_channels=len(args.variables),
        num_classes=len(args.variables),
        latent_dim=2,
        num_filters=[64, 128, 256, 512],
        beta_0=0.0,
        beta_1=0.0,
        beta_2=0.0  # Initialize beta_2 to zero
    ).to(args.device)

    # Initializing the datasets
    dataset_train = cu.climex2torch(
        datadir=args.datadir,
        years=args.years_train,
        variables=args.variables,
        type="lrinterp_to_residuals",
        transfo=True,
        coords=args.coords,
        lowres_scale=args.lowres_scale
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
        dataset_val,
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
    tr_losses_kl2 = {var: [] for var in args.variables}
    val_losses_mae = {var: [] for var in args.variables}
    val_losses_kl = {var: [] for var in args.variables}
    val_losses_kl2 = {var: [] for var in args.variables}



    # initial_beta = 0
    # max_beta = args.beta = 2
    # num_warmup_epochs = 10
    # args.num_epochs = 50

    beta_0 = 1.0
    beta_1 = 0.00
    beta_2 = 0.00
        
    warmup_epochs = 2
    # Training loop
    for epoch in range(1, args.num_epochs + 1):

        probunet_model.beta_0 = beta_0
        probunet_model.beta_1 = beta_1
        probunet_model.beta_2 = beta_2

        # # Increase beta and beta_2 values simultaneously
        # # -----------------------------------------------------------------------------------
        # # Update beta and beta_2 based on the current epoch
        # # probunet_model.beta += beta_increment     # Increment beta
        # probunet_model.beta = 0
        # probunet_model.beta_2 += beta_2_increment # Increment beta_2

        # # Ensure beta and beta_2 do not exceed their maximum values
        # if probunet_model.beta > max_beta:
        #     probunet_model.beta = max_beta
        # if probunet_model.beta_2 > max_beta_2:
        #     probunet_model.beta_2 = max_beta_2
        
        # # -----------------------------------------------------------------------------------

        # # Increasing beta and beta_2 gradually when the loss contains two KL terms
        # # -----------------------------------------------------------------------------------
        # # Update beta and beta_2 based on the current epoch
        # if epoch <= warmup_epochs:
        #     # Warmup phase
        #     probunet_model.beta = 0.0  # Disable KL between posterior and prior
        #     probunet_model.beta_2 += beta_2_increment_first_phase  # Gradually increase beta_2
            
        #     # Ensure beta_2 does not exceed the first phase target
        #     if probunet_model.beta_2 > beta_2_target_first_phase:
        #         probunet_model.beta_2 = beta_2_target_first_phase

        # else:
        #     # Post-warmup phase
        #     probunet_model.beta += beta_increment_second_phase  # Gradually increase beta
        #     probunet_model.beta_2 += beta_2_increment_second_phase  # Continue increasing beta_2
        #     # probunet_model.beta = probunet_model.beta_2  # Set beta equal to beta_2
            
        #     # Ensure beta_2 does not exceed max_beta_2
        #     if probunet_model.beta_2 > max_beta_2:
        #         probunet_model.beta_2 = max_beta_2
        #         # probunet_model.beta = max_beta_2
        #     if probunet_model.beta > max_beta:
        #         probunet_model.beta = max_beta


        # ------------------------------------------------------------------------------------

        # Below is the gradually increasing beta with having it zero for num_warmup_epochs (The loss contains both recon and kl loss)
        # -----------------------------------------------------------------------------------
        # if epoch <= num_warmup_epochs:
        #     # Keep beta at zero for the warmup period
        #     current_beta = 0
        # else:
        #     # Gradually increase beta from 0 to max_beta over the remaining epochs
        #     progress = (epoch - num_warmup_epochs) / (args.num_epochs - num_warmup_epochs)
        #     current_beta = progress * max_beta
        # ----------------------------------------------------------------------------------
        # # Gradually increase beta
        # # current_beta = min(initial_beta + epoch * (max_beta / args.num_epochs), max_beta)

        # # Ensure current_beta does not exceed max_beta
        # current_beta = min(current_beta, max_beta)
        
        # # Set the current beta for the model
        # probunet_model.beta = current_beta

        # --------------------------------------------------------------------------------
        # train the model with just recon loss and keep beta 0 for whole number of epochs
        # probunet_model.beta = 0.0

        print(f"Epoch {epoch}/{args.num_epochs} - beta_0: {probunet_model.beta_0}, beta_1: {probunet_model.beta_1:.4f}, beta_2: {probunet_model.beta_2:.4f}")

        # Training for one epoch
        train_losses_mae, training_losses_kl, training_losses_kl2, kl_div, kl_div2 = tm.train_probunet_step(
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
            tr_losses_kl2[var].append(training_losses_kl2[var])
        
        # Compute average losses for each term
        avg_recon_loss = sum(train_losses_mae.values()) / len(train_losses_mae)  # Average reconstruction loss
        avg_kl_loss = sum(training_losses_kl.values()) / len(training_losses_kl)      # Average KL (posterior vs prior)
        avg_kl2_loss = sum(training_losses_kl2.values()) / len(training_losses_kl2)  # Average KL (posterior vs Gaussian)

        # Ensure losses are scalars by detaching and converting them
        # avg_recon_loss = float(avg_recon_loss.detach().cpu().item())  # Detach and convert to scalar
        avg_kl_loss = float(avg_kl_loss.detach().cpu().item())
        avg_kl2_loss = float(avg_kl2_loss.detach().cpu().item())
        

        if epoch > warmup_epochs:
            # beta_0 = 1.0 / (avg_recon_loss + 1e-7)  
            beta_1 = 1.0 / (avg_kl_loss + 1e-7)
            beta_2 = 1.0 / (avg_kl2_loss + 1e-7)

        else:
            beta_0 = 1.0
            beta_1 = 0.00
            beta_2 = 0.00
        
        
        # Evaluating the model on validation data
        val_losses_mae_running, val_losses_kl_running, val_losses_kl2_running = tm.eval_probunet_model(
            model=probunet_model,
            dataloader=dataloader_val,
            reconstruct=False,
            device=args.device,           
        )
        for var in args.variables:
            val_losses_mae[var].append(val_losses_mae_running[var])
            val_losses_kl[var].append(val_losses_kl_running[var])
            val_losses_kl2[var].append(val_losses_kl2_running[var])
        
        # Visualize the latent space using the training set from multiple batches
        with torch.no_grad():
            num_batches_to_sample = 5  # Number of batches to sample from
            posterior_samples_all = []
            prior_samples_all = []

            train_iter = iter(dataloader_train)
            for _ in range(num_batches_to_sample):
                batch = next(train_iter)
                inputs = batch['inputs'].to(args.device)
                targets = batch['targets'].to(args.device)
                timestamps = batch['timestamps'].unsqueeze(dim=1).to(args.device)

                # Compute posterior and prior distributions
                posterior_dist = probunet_model.posterior(inputs, targets)
                prior_dist = probunet_model.prior(inputs)

                # Sample from them
                posterior_samples = posterior_dist.rsample().cpu().numpy()  # shape: [batch_size, 2]
                prior_samples = prior_dist.rsample().cpu().numpy()          # shape: [batch_size, 2]

                posterior_samples_all.append(posterior_samples)
                prior_samples_all.append(prior_samples)

            # Concatenate all samples from the 5 batches
            posterior_samples_all = np.concatenate(posterior_samples_all, axis=0)
            prior_samples_all = np.concatenate(prior_samples_all, axis=0)

            # Plot the aggregated latent space
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(prior_samples_all[:, 0], prior_samples_all[:, 1], alpha=0.5, label='Prior', color='blue')
            ax.scatter(posterior_samples_all[:, 0], posterior_samples_all[:, 1], alpha=0.5, label='Posterior', color='red')
            ax.set_title(f'Latent Space (Training Data) at Epoch {epoch}')
            ax.legend()
            ax.set_xlabel('z1')
            ax.set_ylabel('z2')

            plt.savefig(f"{args.plotdir}/latent_epoch_{epoch}.png", dpi=300)
            plt.close(fig)
        
        test_batch = next(iter(dataloader_test_random))

        residual_preds, (fig, axs) = tm.sample_residual_probunet_model(
            model=probunet_model,
            dataloader=dataloader_test_random,
            epoch=epoch,
            device=args.device,
            batch=test_batch
        )
        fig.savefig(f"{args.plotdir}/epoch{epoch}_residuals.png", dpi=300)
        plt.close(fig)

        fig_difs, axs_difs = dataset_test.plot_residual_differences(
        residual_preds=residual_preds,
        timestamps_float=test_batch['timestamps_float'][:2],
        epoch=epoch,
        N=2, 
        num_samples=3
        )
        fig_difs.savefig(f"{args.plotdir}/epoch{epoch}_res_difs.png", dpi=300)
        plt.close(fig_difs)

        samples, (fig, axs) = tm.sample_probunet_model(
            model=probunet_model,
            dataloader=dataloader_test_random,
            epoch=epoch,
            device=args.device,
            batch=test_batch
        )
        fig.savefig(f"{args.plotdir}/epoch{epoch}_reconstructed.png", dpi=300)
        plt.close(fig)
    
    # Save losses to a file after training
    losses_to_save = {
        "train_losses_mae": tr_losses_mae,
        "train_losses_kl": tr_losses_kl,
        "train_losses_kl2": tr_losses_kl2,
        "val_losses_mae": val_losses_mae,
        "val_losses_kl": val_losses_kl,
        "val_losses_kl2": val_losses_kl2
    }
    with open(f"{args.plotdir}/losses.pkl", "wb") as f:
        pickle.dump(losses_to_save, f)


    # # Plot training and validation loss curves for each variable
    # plot_losses(tr_losses_mae, tr_losses_kl, val_losses_mae, val_losses_kl, args.variables, args.plotdir)

    plot_losses(tr_losses_mae, tr_losses_kl, tr_losses_kl2, val_losses_mae, val_losses_kl, val_losses_kl2, args.variables, args.plotdir)

