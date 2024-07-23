import torch
import matplotlib.pyplot as plt

import climex_utils as cu
import trainmodel as tm
import networks as nw


if __name__ == "__main__":

    # Importing all required arguments
    args = tm.get_args()

    # Initializing the UNet model
    unet_model = nw.UNet(img_resolution=args.resolution, in_channels=len(args.variables), out_channels=len(args.variables), label_dim=0, use_diffuse=False)
    unet_model.to(args.device)

    # Initiliazing the training and testing dataset
    dataset_train = cu.climexSet(args.datadir, years=args.years_train, coords=args.coords, lowres_scale=args.lowres_scale, train=True)
    dataset_test = cu.climexSet(args.datadir, years=args.years_test, coords=args.coords, lowres_scale=args.lowres_scale, train=False, trainset=dataset_train)

    # Initiliazing the dataloaders
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dataloader_test_random = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initiliazing training objects
    scaler = torch.cuda.amp.GradScaler()
    optimizer = args.optimizer(params=unet_model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    tr_losses = []
    test_losses = []
    # Looping over all epochs
    for step in range(1, args.num_epochs):


        # Training for one epoch (going over all training data)
        epoch_tr_loss = tm.train_step(model=unet_model, dataloader=dataloader_train, loss_fn=loss_fn, optimizer=optimizer, scaler=scaler, step=step, accum=args.accum, device=args.device)
        tr_losses.append(epoch_tr_loss)

        # Evaluating the model on testing data
        epoch_test_loss = tm.eval_model(model=unet_model, dataloader=dataloader_test, loss_fn=loss_fn, device=args.device)
        test_losses.append(epoch_test_loss)

        # Sampling from the model every 5 epochs
        if step % 5 == 0:
            hr_pred, (fig, axs) = tm.sample_model(model=unet_model, dataloader=dataloader_test_random, epoch=step, device=args.device)
            fig.savefig(f"./{args.plotdir}/epoch{step}.png", dpi=300)
            plt.close(fig)

    # Plotting the training and testing losses
    fig = plt.figure(figsize=(15,10))
    plt.plot(tr_losses, lw=2, label='training loss')
    plt.plot(test_losses, lw=2, linestyle='dashed', label='testing loss')
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss for prep, tasmin and tasmax combined")
    plt.legend()
    fig.savefig(f"./{args.plotdir}/loss.png", dpi=300)
    plt.close(fig)
