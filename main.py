import torch
import pandas as pd
import numpy as np

import metrics
import climex_utils as cu
import trainmodel as tm
import deterministic_unet

import os
import time


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
    torch.backends.cudnn.benchmark = False

def run_combination(pipeline, unet, transfo):

    """
    This functions trains a model with a given data pipeline, unet architecture and transformation. 
    It saves the model checkpoint, logs and validation performances in a folder.

    Args:
     - pipeline (str): the data pipeline to use (e.g. "lrinterp_to_residuals")
     - unet (str): the unet architecture to use (e.g. "symmetric")
     - transfo (bool): whether to use the transformation or not
    """

    seed_everything(0)
    args = tm.get_args()

    # if the target is "hr" then we should use 1e-4 for base learning rate for stability
    if pipeline[-2:] == "ls":
        lr = 1e-3
    else:
        lr = 1e-4

    # some combinations are not valid (lr inputs and symmetric unet, hr inputs and asymmetric unet)
    if pipeline[:3] == "lr_" and unet == "symmetric":
        return
    elif pipeline[:8] == "lrinterp" and unet[0] == "a":
        return
    
    print(f"Running {pipeline}-{unet}-{transfo}-{lr}")

    # Initiliazing the training, validation and testing datasets
    dataset_train = cu.climex2torch(args.datadir, years=args.years_subtrain, coords=args.coords, lowres_scale=args.lowres_scale, 
                                    transfo=transfo, type=pipeline, megafile='data_subtrain.nc')
    dataset_earlystop = cu.climex2torch(args.datadir, years=args.years_earlystop, coords=args.coords, lowres_scale=args.lowres_scale,
                                         transfo=transfo, type=pipeline, megafile='data_es.nc')
    dataset_val = cu.climex2torch(args.datadir, years=args.years_val, coords=args.coords, lowres_scale=args.lowres_scale,
                                  transfo=transfo, type=pipeline, megafile='data_val.nc')
    
    # Initializing data loaders
    dataloader_subtrain = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_earlytstop = torch.utils.data.DataLoader(dataset_earlystop, batch_size=args.batch_size, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    # Initializing model
    model = deterministic_unet.UNetAll(type=unet, img_resolution=args.resolution, in_channels=len(args.variables), channel_mult=[1,2,3,4], num_res_blocks=2, 
                                       out_channels=len(args.variables), ds_scale=args.lowres_scale)
    model.to(args.device)

    # Initializing optimizer, loss function, early stopper and learning rate scheduler
    optimizer = args.optimizer(params=model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    loss_fn = torch.nn.L1Loss()
    early_stopper = tm.EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    results = {"training_time": 0,
               "training_epochs": 0,
               "nb_params": sum(param.numel() for param in model.parameters()),
               "temporal_mae": [],
               "spatial_mae": [], 
               "spatial_constraint_violations": []}
    
    tr_losses = {var: [] for var in args.variables}
    es_losses = {var: [] for var in args.variables}
    current_lr = lr
    milestones = []

    # Training the model on subset with early stopping
    for epoch in range(1, 1000):
            
            tr = tm.train_step(model=model, dataloader=dataloader_subtrain, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch, device=args.device)
            es_loss = tm.eval_model(model=model, dataloader=dataloader_earlytstop, reconstruct=False, device=args.device)

            # Saving training and early stopping losses
            for var in args.variables:
                tr_losses[var].append(tr[var])
                es_losses[var].append(es_loss[var])
            
            # Computing scaled early stopping loss
            earlystop_loss = np.array([es_loss["pr"], es_loss["tasmin"], es_loss["tasmax"]])
            if epoch == 1:
                max_loss = earlystop_loss
            es_loss_scaled = earlystop_loss / max_loss
            early_stop, model = early_stopper.early_stop(es_loss_scaled[0]+ 0.5*(es_loss_scaled[1] + es_loss_scaled[2]), model)

            # if early stopping is triggered, we save the number of epochs and break the loop
            if early_stop:
                print("Early stopping at epoch", epoch)
                results["training_epochs"] = epoch - args.patience
                break

            # Learning rate scheduler step with early stopping loss
            lr_scheduler.step(es_loss_scaled[0]+ 0.5*(es_loss_scaled[1] + es_loss_scaled[2]))

            # If learning rate has been updated, save the epoch for later
            if lr_scheduler.get_last_lr()[0] < current_lr:
                current_lr = lr_scheduler.get_last_lr()[0]
                milestones.append(epoch)

            print("Evaluation error: ", es_loss_scaled[0]+ 0.5*(es_loss_scaled[1] + es_loss_scaled[2]))
            print("Learning rate: ", lr_scheduler.get_last_lr()[0])

    # Training the model on the full training set
    training_set = cu.climex2torch(args.datadir, years=args.years_train, coords=args.coords, lowres_scale=args.lowres_scale, transfo=transfo, type=pipeline, megafile='data_train.nc')
    dataloader_train = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    model = deterministic_unet.UNetAll(type=unet, img_resolution=args.resolution, in_channels=len(args.variables), channel_mult=[1,2,3,4], num_res_blocks=2, 
                                       out_channels=len(args.variables), ds_scale=args.lowres_scale)
    model.to(args.device)
    optimizer = args.optimizer(params=model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    loss_fn = torch.nn.L1Loss()
    
    t0 = time.time()
    for e in range(1, epoch - args.patience + 1):
        _ = tm.train_step(model=model, dataloader=dataloader_train, loss_fn=loss_fn, optimizer=optimizer, epoch=e, device=args.device)
        lr_scheduler.step(epoch=e)
        if e in milestones or e == 1:
            print("Learning rate: ", lr_scheduler.get_last_lr()[0])
    results["training_time"] = (time.time() - t0) / 60

    # Evaluating the model on the validation data
    temporal, spatial, violations_count, violations_avg = tm.eval_model(model=model, dataloader=dataloader_val, reconstruct=True, device=args.device, transfo=transfo)
    results["temporal_mae"] = temporal
    results["spatial_mae"] = spatial
    results["spatial_constraint_violations_count"] = violations_count
    results["spatial_constraint_violations_avg"] = violations_avg

    # Saving the model checkpoint and some training logs 
    os.makedirs(f"./results/{pipeline}-{unet}-{transfo}", exist_ok=True)
    torch.save(model.state_dict(), f"./results/{pipeline}-{unet}-{transfo}/model.pt")
    with open(f"./results/{pipeline}-{unet}-{transfo}/logs.txt", "w") as f:
        f.write(str("Training time in minutes: " + str(results["training_time"])))
        f.write("\n")
        f.write(str("Training epochs: " + str(results["training_epochs"])))
        f.write("\n")
        f.write(str("Number of parameters: " + str(results["nb_params"])))
        f.write("\n")

    # Saving the validation performances
    temporal_mae = pd.DataFrame(results["temporal_mae"])
    spatial_mae = pd.DataFrame(results["spatial_mae"])
    spatial_violations_count = pd.DataFrame(results["spatial_constraint_violations_count"])
    spatial_violations_avg = pd.DataFrame(results["spatial_constraint_violations_avg"])
    temporal_mae.to_csv(f"./results/{pipeline}-{unet}-{transfo}/temporal_mae.csv", index=False)
    spatial_mae.to_csv(f"./results/{pipeline}-{unet}-{transfo}/spatial_mae.csv", index=False)
    spatial_violations_count.to_csv(f"./results/{pipeline}-{unet}-{transfo}/spatial_violations_count.csv", index=False)
    spatial_violations_avg.to_csv(f"./results/{pipeline}-{unet}-{transfo}/spatial_violations_avg.csv", index=False)
    
    #_, (fig, axs) = tm.sample_model(model=model, dataloader=dataloader_val, epoch=e, device=args.device)
    #fig.savefig(f"./results/{pipeline}-{unet}-{transfo}/samples.png")
    #plt.close(fig)

    # Saving the power spectrum density of the model's predictions
    hr_pred_psd_pr, hr_pred_psd_tasmin, hr_pred_psd_tasmax = metrics.compute_psd_over_loader(model, dataloader_val, device=args.device)
    hr_pred_psd_pr = pd.Series(hr_pred_psd_pr)
    hr_pred_psd_tasmin = pd.Series(hr_pred_psd_tasmin)
    hr_pred_psd_tasmax = pd.Series(hr_pred_psd_tasmax)
    pred_psd = pd.concat([hr_pred_psd_pr, hr_pred_psd_tasmin, hr_pred_psd_tasmax], axis=1)
    pred_psd.columns = ["pr", "tasmin", "tasmax"]
    pred_psd.to_csv(f"./results/{pipeline}-{unet}-{transfo}/psd_pred_unet.csv", index=False)

    # Saving training and validation losses
    for var in args.variables:
        loss_var = pd.DataFrame({"training": tr_losses[var], "early_stop": es_losses[var]})
        loss_var.to_csv(f"./results/{pipeline}-{unet}-{transfo}/loss_{var}.csv", index=False)


if __name__ == "__main__":

    # Search arguments
    combinations = {"pipeline": ["lrinterp_to_residuals", "lrinterp_to_hr", "lr_to_residuals", "lr_to_hr"],
                    "unet": ["symmetric", "asymmetric_wskips", "asymmetric_woskips"],
                    "transfo": [False, True]}
    
    for pipeline in combinations["pipeline"]:
        for unet in combinations["unet"]:
            for transfo in combinations["transfo"]:
                run_combination(pipeline=pipeline, unet=unet, transfo=transfo)
