import torch
import numpy as np

import codebase.data_utils as du
import codebase.train_utils as tu
import codebase.models.deterministic_unet as detunet
import codebase.models.probabilistic_unet as probunet

if __name__ == "__main__":

    tu.seed_everything(351)
    args = tu.get_args()
     
    dataset_subtrain = du.climex2torch(args.datadir, years=args.years_subtrain, coords=args.coords, lowres_scale=args.lowres_scale, 
                                       transfo=args.transfo, type=args.pipeline)
    dataset_earlystop = du.climex2torch(args.datadir, years=args.years_earlystop, coords=args.coords, lowres_scale=args.lowres_scale,
                                        transfo=args.transfo, type=args.pipeline)
    dataset_megatrain = du.climex2torch(args.datadir, years=args.years_megatrain, coords=args.coords, lowres_scale=args.lowres_scale,
                                        transfo=args.transfo, type=args.pipeline)
    dataset_test = du.climex2torch(args.datadir, years=args.years_test, coords=args.coords, lowres_scale=args.lowres_scale,
                                   transfo=args.transfo, type=args.pipeline)
    
    subtrainloader = torch.utils.data.DataLoader(dataset_subtrain, batch_size=args.batch_size, shuffle=True)
    earlystoploader = torch.utils.data.DataLoader(dataset_earlystop, batch_size=args.batch_size, shuffle=False)
    megatrainloader = torch.utils.data.DataLoader(dataset_megatrain, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # DETERMINISTIC UNET TRAINING

    det_model = detunet.UNet(img_resolution=args.resolution, in_channels=len(args.variables), out_channels=len(args.variables))
    det_model.to(args.device)
    det_model.train()

    optimizer_det = args.optimizer(det_model.parameters(), lr=args.lr)
    lr_scheduler_det = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_det, mode="min", factor=0.1, patience=5)
    loss_fn = torch.nn.L1Loss()
    early_stopper_det = tu.EarlyStopper(patience=args.patience)

    current_lr = args.lr
    milestones = []

    for epoch in range(1, 1000):

        tr_mae = tu.train_step(model=det_model, dataloader=subtrainloader, loss_fn=loss_fn, optimizer=optimizer_det, epoch=epoch, prob=False, device=args.device)
        es_loss = tu.eval_model(model=det_model, dataloader=earlystoploader, prob=False, device=args.device)

        # Computing scaled early stopping loss
        earlystop_loss = np.array([es_loss["pr"], es_loss["tasmin"], es_loss["tasmax"]])
        if epoch == 1:
            max_loss = earlystop_loss
        es_loss_scaled = earlystop_loss / max_loss
        early_stop, model = early_stopper_det.early_stop(es_loss_scaled[0]+ 0.5*(es_loss_scaled[1] + es_loss_scaled[2]), det_model)

        # if early stopping is triggered, break the loop
        if early_stop:
            break

        # Learning rate scheduler step with early stopping loss
        lr_scheduler_det.step(es_loss_scaled[0]+ 0.5*(es_loss_scaled[1] + es_loss_scaled[2]))

        # If learning rate has been updated, save the epoch for later
        if lr_scheduler_det.get_last_lr()[0] < current_lr:
            current_lr = lr_scheduler_det.get_last_lr()[0]
            milestones.append(epoch)

        print("Evaluation error: ", es_loss_scaled[0]+ 0.5*(es_loss_scaled[1] + es_loss_scaled[2]))
        print("Learning rate: ", lr_scheduler_det.get_last_lr()[0])


    optimizer_det = args.optimizer(det_model.parameters(), lr=args.lr)
    lr_scheduler_det = torch.optim.lr_scheduler.MultiStepLR(optimizer_det, milestones=milestones, gamma=0.1)

    for e in range(1, epoch - args.patience + 1):
        tr_mae = tu.train_step(model=det_model, dataloader=megatrainloader, loss_fn=torch.nn.L1Loss(), optimizer=optimizer_det, prob=False, epoch=e, device=args.device)
        lr_scheduler_det.step(epoch=e)
        if e in milestones or e == 1:
            print("Learning rate: ", lr_scheduler_det.get_last_lr()[0])
    

    torch.save(det_model.state_dict(), f"./checkpoints/deterministic_unet_dom6464.pt")

    det_model.eval()
    for i, batch in enumerate(testloader):
        residual = det_model(batch["inputs"].to(args.device), t=batch["timestamps"].unsqueeze(1).to(args.device))
        pred = dataset_test.residual_to_hr(residual.cpu().detach(), batch["lrinterp"])
        if i == 0:
            preds = pred
        else:
            preds = torch.cat((preds, pred), dim=0)
    
    np.save(f"src/codebase/predictions/deterministic_unet_dom6464.npy", preds.numpy())

    # PROBABILISTIC UNET TRAINING

    prob_model = probunet.ProbabilisticUNet(input_channels=len(args.variables), num_classes=len(args.variables),
                                            latent_dim=args.latent_dim, num_filters=args.num_filters)
    prob_model.to(args.device)
    prob_model.train()
    
    optimizer_prob = args.optimizer(prob_model.parameters(), lr=args.lr)
    lr_scheduler_prob = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_prob, mode="min", factor=0.1, patience=5)
    loss_fn = torch.nn.L1Loss()
    early_stopper_prob = tu.EarlyStopper(patience=args.patience)

    current_lr = args.lr
    milestones = []

    for epoch in range(1, 1000):

        if epoch > args.warmup_epochs:
            # beta_0 = 1.0 / (avg_recon_loss + 1e-7)  
            prob_model.beta_1 = 1.0 / (tr_kl + 1e-7)
            prob_model.beta_2 = 1.0 / (tr_kl2 + 1e-7)

        else:
            prob_model.beta_0 = 1.0
            prob_model.beta_1 = 0.00
            prob_model.beta_2 = 0.00

        tr_mae, tr_kl, tr_kl2, _, _ = tu.train_step(model=prob_model, dataloader=subtrainloader, loss_fn=loss_fn, optimizer=optimizer_prob, epoch=epoch, prob=True, device=args.device)
        es_loss = tu.eval_model(model=prob_model, dataloader=earlystoploader, prob=True, device=args.device)

        # Computing scaled early stopping loss
        earlystop_loss = np.array([es_loss["pr"], es_loss["tasmin"], es_loss["tasmax"]])
        if epoch == 1:
            max_loss = earlystop_loss
        es_loss_scaled = earlystop_loss / max_loss
        early_stop, model = early_stopper_prob.early_stop(es_loss_scaled[0]+ 0.5*(es_loss_scaled[1] + es_loss_scaled[2]), prob_model)

        # if early stopping is triggered, break the loop
        if early_stop:
            break

        # Learning rate scheduler step with early stopping loss
        lr_scheduler_prob.step(es_loss_scaled[0]+ 0.5*(es_loss_scaled[1] + es_loss_scaled[2]))

        # If learning rate has been updated, save the epoch for later
        if lr_scheduler_prob.get_last_lr()[0] < current_lr:
            current_lr = lr_scheduler_prob.get_last_lr()[0]
            milestones.append(epoch)

        print("Evaluation error: ", es_loss_scaled[0]+ 0.5*(es_loss_scaled[1] + es_loss_scaled[2]))
        print("Learning rate: ", lr_scheduler_prob.get_last_lr()[0])

    
    prob_model = probunet.ProbabilisticUNet(input_channels=len(args.variables), num_classes=len(args.variables),
                                            latent_dim=args.latent_dim, num_filters=args.num_filters)
    prob_model.to(args.device)
    prob_model.train()

    optimizer_prob = args.optimizer(prob_model.parameters(), lr=args.lr)
    lr_scheduler_prob = torch.optim.lr_scheduler.MultiStepLR(optimizer_prob, milestones=milestones, gamma=0.1)

    for e in range(1, epoch - args.patience + 1):
        if e > args.warmup_epochs:
            # beta_0 = 1.0 / (avg_recon_loss + 1e-7)  
            prob_model.beta_1 = 1.0 / (tr_kl + 1e-7)
            prob_model.beta_2 = 1.0 / (tr_kl2 + 1e-7)

        else:
            prob_model.beta_0 = 1.0
            prob_model.beta_1 = 0.00
            prob_model.beta_2 = 0.00
        tr_mae, tr_kl, tr_kl2, _, _ = tu.train_step(model=prob_model, dataloader=megatrainloader, loss_fn=torch.nn.L1Loss(), optimizer=optimizer_prob, prob=True, epoch=e, device=args.device)
        lr_scheduler_prob.step(epoch=e)
        if e in milestones or e == 1:
            print("Learning rate: ", lr_scheduler_prob.get_last_lr()[0])
    

    torch.save(prob_model.state_dict(), f"src/checkpoints/probabilistic_unet_dom6464.pt")

    prob_model.eval()
    for i, batch in enumerate(testloader):
        for s in range(args.num_samples):
            residual = prob_model(batch["inputs"].to(args.device), t=batch["timestamps"].unsqueeze(1).to(args.device), training=False)
            pred = dataset_test.residual_to_hr(residual.cpu().detach(), batch["lrinterp"])
            if s == 0:
                preds_ = pred.unsqueeze(1)
            else:
                preds_ = torch.cat((preds_, pred.unsqueeze(1)), dim=1)
        if i == 0:
            preds = preds_
        else:
            preds = torch.cat((preds, preds_), dim=0)
    
    np.save(f"src/codebase/predictions/probabilistic_unet_dom6464.npy", preds.numpy())
    