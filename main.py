import torch
import wandb
import numpy as np
import pytorch_warmup as warmup
import matplotlib.pyplot as plt

import metrics
import climex_utils as cu
import trainmodel as tm
import models as mdls

import lr_to_hr_unet 
import deterministic_unet

import pandas as pd

# For plotting the smoothed training and validation losses
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# For reproducibility
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
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    # Importing all required arguments
    args = tm.get_args()
    
    seed_everything(32)

    # Initializing WandB
    if args.wandb:
        wandb.init(project="prob-unet-mds", config={"model": args.ds_model, "epochs": 1}, name=f"{args.resolution}_{args.ds_model}")

    # Initiliazing the training, validation and testing datasets
    dataset_train = cu.climex2torch(args.datadir, years=args.years_train, coords=args.coords, lowres_scale=args.lowres_scale, standardization="perpixel", transfo=False, type="lr_to_residuals")
    dataset_val = cu.climex2torch(args.datadir, years=args.years_val, coords=args.coords, lowres_scale=args.lowres_scale, standardization="perpixel", transfo=False, type="lr_to_residuals")
    dataset_test = cu.climex2torch(args.datadir, years=args.years_test, coords=args.coords, lowres_scale=args.lowres_scale, standardization="perpixel", transfo=False, type="lr_to_residuals")

    # Initializing model
    model = lr_to_hr_unet.LR_to_HR_UNet_v2(input_resolution=args.resolution[0]//args.lowres_scale, in_channels=len(args.variables), ds_scale=args.lowres_scale, 
                                           num_res_blocks=2, channels_mult=[1, 2, 3, 4, 5], out_channels=len(args.variables))
    #model = deterministic_unet.UNet(img_resolution=args.resolution, in_channels=len(args.variables), out_channels=len(args.variables), channel_mult=[1, 2, 3, 4], num_blocks=2)
    print("Model's number of parameters:", sum(param.numel() for param in model.parameters()))
    model.to(args.device)
    if args.wandb:
        wandb.watch(models=model)

    # Initiliazing the dataloaders
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    dataloader_val_random = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    num_train_steps = len(dataloader_train) * args.num_epochs

    # Initiliazing training objects
    optimizer = args.optimizer(params=model.parameters(), lr=args.lr)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    #warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    loss_fn = torch.nn.L1Loss()
    early_stopper = tm.EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    tr_losses = {var: [] for var in args.variables}
    val_losses = {var: [] for var in args.variables}
    # Looping over all epochs
    for epoch in range(1, 1000):

        # Training for one epoch (going over all training data)
        tr_running_losses = tm.train_step(model=model, dataloader=dataloader_train, loss_fn=loss_fn, optimizer=optimizer, 
                                          epoch=epoch, device=args.device)
        #lr_scheduler.step()
        
        # Evaluating the model on validation data
        val_running_losses = tm.eval_model(model=model, dataloader=dataloader_val, loss_fn=loss_fn, reconstruct=False, device=args.device)

        # Saving the training and validation losses
        for var in args.variables:
            tr_losses[var].append(tr_running_losses[var])
            val_losses[var].append(val_running_losses[var])

        early_stop, model = early_stopper.early_stop(val_losses["pr"][-1], model)
        if early_stop:
            print("Early stopping at epoch", epoch)
            break


    # Sampling and plotting the model's predictions
    hr_pred, (fig, axs) = tm.sample_model(model=model, dataloader=dataloader_val_random, epoch=epoch-5, device=args.device)
    fig.savefig(f"./{args.plotdir}/epoch{epoch-5}_samples_from_unet1.png", dpi=300)
    plt.close(fig)

    # Plotting the power spectrum density of the model's predictions
    hr_pred_psd_pr, hr_pred_psd_tasmin, hr_pred_psd_tasmax = metrics.compute_psd_over_loader(model, dataloader_val, device=args.device)
    hr_pred_psd_pr = pd.Series(hr_pred_psd_pr)
    hr_pred_psd_tasmin = pd.Series(hr_pred_psd_tasmin)
    hr_pred_psd_tasmax = pd.Series(hr_pred_psd_tasmax)
    pred_psd = pd.concat([hr_pred_psd_pr, hr_pred_psd_tasmin, hr_pred_psd_tasmax], axis=1)
    pred_psd.columns = ["pr", "tasmin", "tasmax"]
    pred_psd.to_csv(f"./psd_pred_unet.csv", index=False)
    

    # Plotting training and validation losses
    for var in args.variables:
        x = np.arange(1, epoch+1)
        fig = plt.figure(figsize=(15,10))
        plt.plot(x, tr_losses[var], lw=2, label='training loss')
        plt.plot(x, val_losses[var], lw=2, linestyle='dashed', label='validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("MAE Loss")
        plt.title(f"MAE Loss for {var}")
        plt.legend()
        fig.savefig(f"./{args.plotdir}/loss_{var}.png", dpi=300)
        plt.close(fig)

        torch.save(model.state_dict(), f"./{args.checkpoints_dir}/{args.ds_model}.pt")
        torch.save(optimizer.state_dict(), f"./{args.checkpoints_dir}/{args.ds_model}_optimizer.pt")

        # Evaluating the model on the validation data
        mae_loss = tm.eval_model(model=model, dataloader=dataloader_val, loss_fn=torch.nn.L1Loss(), reconstruct=True, device=args.device)
        print("MAE for precipitation on validation data: ", np.mean(mae_loss["pr"]))
        print("MAE for tasmin on validation data: ", np.mean(mae_loss["tasmin"]))
        print("MAE for tasmax on validation data: ", np.mean(mae_loss["tasmax"]))
