import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from cartopy import crs as ccrs

import codebase.data_utils as du
import codebase.train_utils as tu
import codebase.metrics as metrics
import codebase.models.bcsd as bcsd
import codebase.models.linearregression as lr

if __name__ == "__main__":

    # -------------------------- #
    #          Load data         #

    args = tu.get_args()

    trainset = du.climex2torch(datadir=args.datadir, years=args.years_megatrain, coords=args.coords, type="lrinterp_to_residuals", lowres_scale=16, transfo=True, megafile="data/data_megatrain.nc")
    testset = du.climex2torch(datadir=args.datadir, years=args.years_test, coords=args.coords, type="lrinterp_to_residuals", lowres_scale=16, transfo=True, megafile="data/data_test.nc")
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    lat, lon = testset.data.lat[0].load().data, testset.data.lon[0].load().data

    # -------------------------- #
    #      Load predictions      #

    test_hr = testset.hr
    test_lr = torch.nn.AvgPool2d(args.lowres_scale)(test_hr)

    test_nninterp = torch.nn.functional.interpolate(test_lr, scale_factor=args.lowres_scale, mode="nearest")
    test_bcsd = bcsd.BCSD(trainset, testset)
    test_linearregression = lr.get_lr_preds()["test_preds"]
    test_probunet = torch.from_numpy(np.load("codebase/predictions/probabilistic_unet.npy"))
    test_detunet = torch.from_numpy(np.load("codebase/predictions/deterministic_unet.npy"))

    preds_list = [test_nninterp, test_bcsd, test_linearregression, test_detunet, test_probunet]
    list_names = ["1nn", "bcsd", "linear regression", "deterministic unet", "probabilistic unet"]
    colors = ["teal", "magenta", "purple", "crimson", "orange"]
    styles = ["-", "--", "-.", "--", "-"]
    #markers = [">", "*", "X", "<", "o"]

    title_log = "posterior+gaussian_prior+posterior_131fcomb"
    os.makedirs(f"out/{title_log}", exist_ok=True)

    # -------------------------- #
    #   Compute and log metrics  #

    # PSD

    test_bcinterp = torch.nn.functional.interpolate(test_lr, scale_factor=args.lowres_scale, mode="bicubic") # instead of NN interp for plot readibility
    psd_list_names =["bicubic"] + list_names[1:]
    psd_preds_list = [test_bcinterp] + preds_list[1:]

    psd = {var: [] for var in args.variables}
    psd_ref = metrics.compute_psd_over_groundtruth(test_hr, transfo=True)
    for preds, name in zip(psd_preds_list, psd_list_names):
        if name == "bcsd":
            psd_i = metrics.compute_psd_over_groundtruth(preds, transfo=False)
        else:
            psd_i = metrics.compute_psd_over_groundtruth(preds, transfo=True)
            if name == "probabilistic_unet":
                psd_ens = []
                psd_std_low = {var: [] for var in args.variables}
                psd_std_high = {var: [] for var in args.variables}
                for s in range(args.num_samples):
                    psd_ens.append(metrics.compute_psd_over_groundtruth(preds[:, s], transfo=True))
                psd_ens = np.stack(psd_ens, axis=1)
                for v, var in enumerate(args.variables):
                    for wn in range(psd_ens[v].shape[1]):
                        confint_wn = bootstrap(psd_ens[v, :, wn], np.std, n_resamples=1000, confidence_level=0.95).confidence_interval
                        psd_std_low[var].append(confint_wn[0])
                        psd_std_high[var].append(confint_wn[1])

        for v, var in enumerate(args.variables):
            psd[var].append(psd_i[v])

    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    axbis = []
    for i, ax in enumerate(axs):
        max = 0
        axbis.append(ax.twiny())
        scale_km = np.arange(0, 65, 1)*2*11
        ax.plot(psd_ref[i], color="black", linestyle="-", lw=2.5, label="ground truth")
        axbis[-1].plot(scale_km, np.zeros_like(scale_km), alpha=0)
        for psd_i, name, color, style in zip(psd[args.variables[i]], psd_list_names, colors, styles):
            ax.plot(psd_i, color=color, linestyle=style, lw=2.5, label=name)
            if name == "probabilistic_unet":
                ax.fill_between(np.arange(psd_i.shape[0]), psd_std_low[args.variables[i]], psd_std_high[args.variables[i]], color=color, alpha=0.3)
        ax.text(0.8, 0.92, args.variables[i], fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.set_xlabel("Zonal wavenumber")
        ax.set_yscale('log')
        ax.set_xscale('log')
        axbis[-1].set_xlabel("Approximate scale (km)")
        axbis[-1].set_xscale("log")
        axbis[-1].invert_xaxis()
    axs[0].set_ylim(top=2.5e9)
    axs[1].set_ylim(top=4e9)
    axs[2].set_ylim(top=2.5e9)
    axs[0].set_ylabel("Mean Power (dB)")
    axs[2].legend(loc="lower left", fontsize=14)
    fig.savefig(f"out/{title_log}/psd.png")

    # MAE, RMSE and CRPS

    test_hr = du.inv_transform(test_hr)
    test_nninterp = du.inv_transform(test_nninterp)
    test_linearregression = du.inv_transform(test_linearregression)
    test_detunet = du.inv_transform(test_detunet)
    test_probunet = du.inv_transform(test_probunet, prob=True)

    test_bcsd[:, 0] = du.kgm2sTommday(test_bcsd[:, 0])
    test_bcsd[:, 1] = du.KToC(test_bcsd[:, 1])
    test_bcsd[:, 2] = du.KToC(test_bcsd[:, 2])

    mae, rmse = [], []
    for preds in preds_list:
        if len(preds.shape) == 5:
            ens_mae = torch.abs(test_hr.unsqueeze(1) - preds).mean(dim=(0, 3, 4)).numpy()
            ens_rmse = torch.sqrt(((test_hr.unsqueeze(1) - preds) ** 2).mean(dim=(0, 3, 4))).numpy()
            std_mae = np.std(ens_mae, axis=0)
            std_rmse = np.std(ens_rmse, axis=0)
            preds = preds.mean(dim=1)
        mae.append(torch.abs(test_hr - preds).mean(dim=(0, 2, 3)).numpy())
        rmse.append(torch.sqrt(((test_hr - preds) ** 2).mean(dim=(0, 2, 3))).numpy())
    crps_probunet = metrics.crps_over_groundtruth(test_hr, test_probunet)

    f = open(f"out/{title_log}/reduced_metrics.txt", "w")
    f.write(f"MAE\n")
    for i, name in enumerate(list_names):
        if name == "probabilistic unet":
            f.write(f"{name}: {mae[i]} +/- {std_mae}\n")
        else:
            f.write(f"{name}: {mae[i]}\n")
    f.write(f"\nRMSE\n")
    for i, name in enumerate(list_names):
        if name == "probabilistic unet":
            f.write(f"{name}: {rmse[i]} +/- {std_rmse}\n")
        else:
            f.write(f"{name}: {rmse[i]}\n")
    f.write(f"\nCRPS\n")
    f.write(f"probabilistic unet: {crps_probunet}\n")
    f.close()

    q_mae, q_rmse = metrics.quantile_mae_rmse(test_hr, preds_list, list_names)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for i, ax in enumerate(axs):
        for name, color, style in zip(list_names, colors, styles):
            ax.plot(np.arange(0, 1., 0.01), q_mae[name][i], label=name, linestyle=style, color=color, lw=2.5)
        ax.set_title(["pr (mm/day)", "tasmin (°C)", "tasmax (°C)"][i])
        ax.set_xlabel("Quantile")
        ax.set_xticks([.2, .4, .6, .8])
        if i == 0:
            ax.set_xlim(0.9, 0.995)
            ax.set_ylim(2, 23)
            ax.set_xticks([.92, .94, .96, .98])
    axs[0].set_ylabel("MAE")
    axs[0].legend(loc="upper left", fontsize=14)
    fig.savefig(f"out/{title_log}/quantile_mae.png")

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for i, ax in enumerate(axs):
        for name, color, style in zip(list_names, colors, styles):
            ax.plot(np.arange(0, 1., 0.01), q_rmse[name][i], label=name, linestyle=style, color=color, lw=2.5)
        ax.set_title(["pr (mm/day)", "tasmin (°C)", "tasmax (°C)"][i])
        ax.set_xlabel("Quantile")
        ax.set_xticks([.2, .4, .6, .8])
        if i == 0:
            ax.set_xlim(0.9, 0.995)
            ax.set_ylim(2, 30)
            ax.set_xticks([.92, .94, .96, .98])
    axs[0].set_ylabel("RMSE")
    axs[0].legend(loc="upper left", fontsize=14)
    fig.savefig(f"out/{title_log}/quantile_rmse.png")

    # Distribution histograms

    ranges = []
    for v, var in enumerate(args.variables):
        min_v, max_v = test_hr[:, v].min(), test_hr[:, v].max()
        ranges.append(np.linspace(min_v, max_v, 100))

    fig, axs = plt.subplots(1, 3, figsize=(20, 4), constrained_layout=True)
    for i, (range, var) in enumerate(zip(ranges, ["pr (mm/day)", "tmin (°C)", "tmax (°C)"])):
        hist = np.histogram(test_hr[:, i].flatten(), bins=range)
        axs[i].stairs(np.log(hist[0]), hist[1], color="silver", fill=True)
        axs[i].stairs(np.log(hist[0]), hist[1], label="ground truth", color="black", lw=2.5)
        for preds, name, color in zip(preds_list, list_names, colors):
            if len(preds.shape) == 5:
                hist = np.histogram(preds[:, :, i].mean(dim=1).flatten(), bins=range)
            else:
                hist = np.histogram(preds[:, i].flatten(), bins=range)
            axs[i].stairs(np.log(hist[0]), hist[1], label=name, lw=2.5, color=color)        
        axs[i].set_ylabel("Log-Freq")
        axs[i].set_xlabel(var)
    axs[0].legend(loc="upper right", fontsize=14)
    fig.savefig(f"out/{title_log}/distribution_histograms.png")

    # Spread-Skill Ratio 

    ssr_probunet = metrics.spread_skill_ratio(test_hr, test_probunet, reduction_dims=(0, 2, 3))
    f = open(f"out/{title_log}/reduced_metrics.txt", "a")
    f.write(f"\nSSR\n")
    f.write(f"probabilistic unet: {ssr_probunet}\n")
    f.close()

    spatial_ssr_probunet = metrics.spread_skill_ratio(test_hr, test_probunet, reduction_dims=0)

    # Initializing Plate Carrée and Rotated Pole projections (for other projections see https://scitools.org.uk/cartopy/docs/latest/reference/crs.html)
    rotatedpole_prj = ccrs.RotatedPole(pole_longitude=83.0, pole_latitude=42.5)
    platecarree_proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': rotatedpole_prj})
    for i, ax in enumerate(axs):
        ax.coastlines()
        gl = ax.gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        if i>0:
            gl.left_labels = False
            vmin, vmax = - np.abs(spatial_ssr_probunet[1:]).max(), np.abs(spatial_ssr_probunet[1:]).max()
        else:
            vmin, vmax = - np.abs(spatial_ssr_probunet[0]).max(), np.abs(spatial_ssr_probunet[0]).max()
        im = ax.pcolormesh(lon, lat, spatial_ssr_probunet[i], transform=platecarree_proj, cmap="PiYG", vmin=vmin, vmax=vmax)
        ax.set_title(args.variables[i])
        plt.colorbar(im, extend="both", shrink=0.75)
    fig.savefig(f"out/{title_log}/spatial_ssr.png")

    # SAL

    f = open(f"out/{title_log}/reduced_metrics.txt", "a")
    f.write(f"\nSAL\n")
    for preds, name in zip(preds_list, list_names):
        sal = metrics.sal(preds, test_hr)
        f.write(f"{name}: {sal}\n")
    f.close()



