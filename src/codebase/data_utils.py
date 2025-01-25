import glob
from dask.distributed import Client
import xarray as xr
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cartopy import crs as ccrs

import torch
import torch.nn as nn
from torch.utils.data import Dataset

#####

# Converts np.datetime64 to np.float64 (number of days to date)
def date_to_float(date_array):
    return date_array.values.astype(float)

# Converts np.float64 to np.datetime64
def float_to_date(float_array):
    return np.array(float_array, dtype="datetime64[ns]")

# For precipitation
def kgm2sTommday(data):
    return data*24*60*60

# For inverse transformation
def softplus_inv(data, threshold=20., c=1e-7):
    mask = data > threshold
    data[mask] = data[mask]
    data[~mask] = torch.log(torch.exp(data[~mask] + c) - 1.)
    return data

def softplus(data, threshold=20., c=1e-7):
    mask = data > threshold
    data[mask] = data[mask]
    data[~mask] = torch.log(torch.exp(data[~mask]) + 1.) - c
    return data
        
# For temperature
def KToC(data):
    return data - 273.15


def inv_transform(data, prob=False):
    if prob:
        data[:, :, 0] = softplus(data[:, :, 0])
        data[:, :, 2] = softplus(data[:, :, 2], c=0) + data[:, :, 1]
        data[:, :, 0] = kgm2sTommday(data[:, :, 0])
        data[:, :, 1] = KToC(data[:, :, 1])
        data[:, :, 2] = KToC(data[:, :, 2])
    else:
        data[:, 0] = softplus(data[:, 0])
        data[:, 2] = softplus(data[:, 2], c=0) + data[:, 1]
        data[:, 0] = kgm2sTommday(data[:, 0])
        data[:, 1] = KToC(data[:, 1])
        data[:, 2] = KToC(data[:, 2])
    return data

####

class climex2torch(Dataset):

    """
    Dataset class that loads and converts data from NetCDF files to a Pytorch tensor on initialization. 
    climex2torch object can be fed to a Pytorch Dataloader.
    """

    def __init__(self, datadir, years=range(1960, 2020), variables=["pr", "tasmin", "tasmax"], coords=[120, 184, 120, 184], type="lr_to_hr", lowres_scale = 4, transfo=False, megafile=None):

        """
        datadir: (str) path to the directory containing NetCDF files;
        years: (list of int) indicates which years should climex2torch import data from;
        variables: (list of str) indicates what variables should climex2torch import data from;
        coords: (list of int) (form: [start_rlon, end_rlon, start_rlat, end_rlat]) climex2torch will only import data from the resulting window;
        type: (str) indicates the data pipeline to train the model on (lr_to_hr, lr_to_residuals, lrinterp_to_residuals, lrinterp_to_hr);
        lowres_scale: (int) downscaling factor;
        """

        super().__init__()

        # Setup dask distributed cluster
        client = Client()

        self.datadir = datadir
        self.years = years
        self.variables = variables
        self.nvars = len(variables)
        self.coords = coords
        self.type = type
        self.lowres_scale = lowres_scale
        self.transfo = transfo
        self.megafile = megafile
        self.epsilon = 1e-10 #used for standardization
        self.lrstats = None #used for standardization

        # Preprocessing function to select only desired coordinates
        def select_coords(ds):
            return ds.isel(rlon=slice(coords[0], coords[1]), rlat=slice(coords[2],coords[3]))
        
        if megafile is None:

            # Recursively getting all NetCDF files names
            files = []
            for year in self.years:
                for var in variables:
                    files.append(glob.glob("{path}/*_{var}_*_{year}_*".format(path=self.datadir, var=var, year=year))[0])  

            print("Opening and lazy loading netCDF files")  

            # Importing all NetCDF files into a xarray Dataset with lazy loading
            self.data = xr.open_mfdataset(paths=files, engine='h5netcdf', preprocess=select_coords, data_vars="minimal", coords="minimal", compat="override", parallel=True)[self.variables]

        else:

            print("Opening and lazy loading megafile")
            self.data = xr.open_dataset(self.megafile)[self.variables]
        
        # Extracting latitude and longitude data (for plotting function) and timestamps
        self.lon = self.data.lon
        self.lat = self.data.lat

        # Extracting time features
        time = self.data.indexes["time"].to_datetimeindex()
        month = np.sin(2*np.pi*time.month/12)
        day = np.cos(2*np.pi*time.day/31)
        self.timestamps = torch.from_numpy(np.array(month + day)).float()
        self.timestamps_float = date_to_float(time)

        data_temp = self.data

        # Dropping unnecessary variables and encoding
        data_temp = data_temp.drop_vars(["lat", "lon"]).drop_indexes(["rlon", "rlat"]).drop_encoding().to_array()

        print("Loading dataset into memory")
        data_temp.load()

        print("Converting xarray Dataset to Pytorch tensor")

        # Loading into memory high-resolution ground-truth data from desired spatial window and converting to Pytorch tensor (time, nvar, height, width)
        self.hr = torch.from_numpy(data_temp.to_numpy()).transpose(0, 1)

        # Tranformations (prep > 0 and tmax > tmin)
        if self.transfo:
            self.hr[:, 0, :, :] = softplus_inv(self.hr[:, 0, :, :])
            self.hr[:, 2, :, :] = softplus_inv(self.hr[:, 2, :, :] - self.hr[:, 1, :, :], c=0.)

        client.close()

        print("")
        print("##########################################")
        print("############ PROCESSING DONE #############")
        print("##########################################")
        print("")


    def __len__(self):
         return len(self.timestamps)

    def __getitem__(self, idx):

        if self.type == "lr_to_hr":

            hr = self.hr[idx]
            lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(self.hr[idx])

            # If standardization statistics are not computed yet, compute them
            if self.lrstats is None :
                print("Computing statistics for standardization")
                self.lrstats = self.compute_stats()

            lr_stand = (lr - self.lrstats[0][0]) / (self.lrstats[0][1] + self.epsilon)
            hr_stand = (hr - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)

            return {"inputs": lr_stand,
                    "targets": hr_stand,
                    "timestamps": self.timestamps[idx],
                    "timestamps_float": self.timestamps_float[idx],
                    "hr": hr, 
                    "lr": lr,
                    "lrinterp": nn.functional.interpolate(input=lr.unsqueeze(0), scale_factor=self.lowres_scale).squeeze()}
        
        if self.type == "lr_to_residuals":

            hr = self.hr[idx]
            lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(self.hr[idx])

            # If standardization statistics are not computed yet, compute them
            if self.lrstats is None :
                print("Computing statistics for standardization")
                self.lrstats = self.compute_stats()

            lr_stand = (lr - self.lrstats[0][0]) / (self.lrstats[0][1] + self.epsilon)
            hr_stand = (hr - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)

            residual = hr_stand - nn.functional.interpolate(input=lr_stand.unsqueeze(0), scale_factor=self.lowres_scale).squeeze()

            return {"inputs": lr_stand,
                    "targets": residual,
                    "timestamps": self.timestamps[idx],
                    "timestamps_float": self.timestamps_float[idx],
                    "hr": hr, 
                    "lr": lr,
                    "lrinterp": nn.functional.interpolate(input=lr.unsqueeze(0), scale_factor=self.lowres_scale).squeeze()}
        
        elif self.type == "lrinterp_to_residuals":

            hr = self.hr[idx]

            # Low-resolution data is obtained by averaging the high-resolution data and then upsampling it
            lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(hr)
            lrinterp = nn.functional.interpolate(input=lr.unsqueeze(0), scale_factor=self.lowres_scale).squeeze() 

            # If standardization statistics are not computed yet, compute them
            if self.lrstats is None :
                print("Computing statistics for standardization")
                self.lrstats = self.compute_stats()

            lrinterp_stand = (lrinterp - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)
            hr_stand = (hr - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)

            residual = hr_stand - lrinterp_stand
            timestamp = self.timestamps[idx]

            return {"inputs": lrinterp_stand,
                    "targets": residual,
                    "timestamps": timestamp,
                    "timestamps_float": self.timestamps_float[idx],
                    "hr": hr, 
                    "lr": lr,
                    "lrinterp": lrinterp}

        elif self.type == "lrinterp_to_hr":

            hr = self.hr[idx]

            # Low-resolution data is obtained by averaging the high-resolution data and then upsampling it
            lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(hr)
            lrinterp = nn.functional.interpolate(input=lr.unsqueeze(0), scale_factor=self.lowres_scale).squeeze() 

            # If standardization statistics are not computed yet, compute them
            if self.lrstats is None :
                print("Computing statistics for standardization")
                self.lrstats = self.compute_stats()

            lrinterp_stand = (lrinterp - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)
            hr_stand = (hr - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon) 

            timestamp = self.timestamps[idx]

            return {"inputs": lrinterp_stand,
                    "targets": hr_stand,
                    "timestamps": timestamp,
                    "timestamps_float": self.timestamps_float[idx],
                    "hr": hr, 
                    "lr": lr,
                    "lrinterp": lrinterp}


    # Computes the statistics of the low-resolution data for standardization
    def compute_stats(self, linear_regression=False):

        if linear_regression:
            lr = self.lrpreds
        else:
            lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(self.hr)

        mean, std = lr.mean(dim=0), lr.std(dim=0) 
        # Extend the dimension to match high-resolution
        if linear_regression:
            mean_hrdim = None
            std_hrdim = None
        else:
            mean_hrdim = mean.repeat_interleave(repeats=self.lowres_scale, dim=1).repeat_interleave(repeats=self.lowres_scale, dim=2)
            std_hrdim = std.repeat_interleave(repeats=self.lowres_scale, dim=1).repeat_interleave(repeats=self.lowres_scale, dim=2)

        return (mean, std), (mean_hrdim, std_hrdim)

    # Computes the inverse of the standardization for the residual
    def invstand_residual(self, standardized_residual, linear_regression=False):
        if linear_regression:
            return standardized_residual * (self.lrstats[0][1] + self.epsilon)
        elif self.type == "lr_to_hr" or self.type == "lrinterp_to_hr":
            return standardized_residual * (self.lrstats[1][1] + self.epsilon) + self.lrstats[1][0]
        elif self.type == "lrinterp_to_residuals" or self.type == "lr_to_residuals":
            return standardized_residual * (self.lrstats[1][1] + self.epsilon)
    
    # Adds the predicted residual to the input upsampled high-resolution
    def residual_to_hr(self, residual, lrinterp, linear_regression=False):
        return lrinterp + self.invstand_residual(residual, linear_regression)
    
    # Plot a batch (N) of samples (upsampled low-resolution, predicted high-resolution, groundtruth high-resolution)
    def plot_batch(self, lrinterp, hr_preds, hr, timestamps, epoch, N=2):

        """
        Plots low-resolution inputs, multiple high-resolution predictions, and ground truth high-resolution outputs.

        Parameters:
        - lrinterp (torch.Tensor): Interpolated low-resolution inputs of shape [N, nvars, H, W].
        - hr_preds (torch.Tensor): Predicted high-resolution outputs of shape [N, num_samples, nvars, H, W].
        - hr (torch.Tensor): Ground truth high-resolution outputs of shape [N, nvars, H, W].
        - timestamps (torch.Tensor): Timestamps corresponding to each sample.
        - epoch (int): Current epoch number, used for plot titles.
        - N (int): Number of samples to plot (default is 2).

        Returns:
        - fig: The matplotlib figure object containing the plots.
        - axs: The axes of the plots for further customization if needed.
        """

        # Initializing Plate Carrée and Rotated Pole projections (for other projections see https://scitools.org.uk/cartopy/docs/latest/reference/crs.html)
        rotatedpole_prj = ccrs.RotatedPole(pole_longitude=83.0, pole_latitude=42.5)
        platecarree_proj = ccrs.PlateCarree()

        if len(hr_preds.shape) == 3:
            hr_preds = hr_preds.unsqueeze(1)
        num_samples = hr_preds.shape[1]
        total_cols= num_samples + 3

        # Initializing figure and subfigures (one subfigure per date)
        fig = plt.figure(figsize=(N * 18, 12), constrained_layout=True)
        subfigs = fig.subfigures(1, N, wspace=0.05)

        # Different colormaps for different type of climate variables
        prep_colors = [
            (1., 1., 1.), 
            (0.5, 0.88, 1.),
            (0.1, 0.15, 0.8),
            (0.39, 0.09, 0.66), 
            (0.85, 0.36, 0.14),
            (0.99, 0.91, 0.3)
        ]
        prep_colormap = mpl.colors.LinearSegmentedColormap.from_list(name="prep", colors=prep_colors)
        cmaps = {'pr': prep_colormap, 'temp': cm.get_cmap('RdBu_r'), 'error': cm.get_cmap('gist_heat_r')}

        axs = []
        # Batch (N) plotting loop      
        for j in range(N):

            axs.append(subfigs[j].subplots(self.nvars, total_cols, subplot_kw={'projection': rotatedpole_prj}, gridspec_kw={'wspace':0.01, 'hspace':0.005}))

            # Extracting latitude and longitude data corresponding to the j-th sample from the batch
            lat, lon = self.lat.sel(time=str(float_to_date(timestamps[j]))[:10]).load().to_numpy().squeeze(), self.lon.sel(time=str(float_to_date(timestamps[j]))[:10]).load().to_numpy().squeeze()

            # Variables plotting loop
            temp_max_abs = []
            temp_ims = []
            for i in range(self.nvars):

                if self.variables[i] == "pr":

                    cmap = cmaps["pr"]
                    unit = " (mm/day)"

                    # Converting units in mm/day and computing scaling values for colormap
                    if self.transfo:
                        lr_sample = kgm2sTommday(softplus(lrinterp[j,i]))
                        hr_preds_samples = [kgm2sTommday(softplus(hr_preds[j, s, i])) for s in range(num_samples)]
                        hr_sample = kgm2sTommday(softplus(hr[j,i]))
                    else:
                        lr_sample = kgm2sTommday(lrinterp[j,i])
                        hr_preds_samples = [kgm2sTommday(hr_preds[j, s, i]) for s in range(num_samples)]
                        hr_sample = kgm2sTommday(hr[j,i])
                    vmin, vmax = 0, max(torch.amax(lr_sample), torch.amax(hr_preds_samples), torch.amax(hr_sample))

                    # Computing absolute error and setting corresponding vmin, vmax
                    error_sample = torch.abs(hr_sample - torch.mean(torch.stack(hr_preds_samples), dim=0)) 
                    err_vmin, err_vmax = 0, torch.amax(error_sample)

                    # Setting cartopy features on the Axes objects
                    for l in range(total_cols):
                        axs[j][i, l].coastlines()
                        gl = axs[j][i, l].gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                        gl.top_labels = False
                        gl.right_labels = False
                        if l > 0: 
                            gl.left_labels = False

                    # Plotting samples in the following order: upsampled low-resolution, predicted high-resolution, groundtruth high-resolution, error
                    axs[j][i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    for s in range(num_samples):
                        axs[j][i, s+1].pcolormesh(lon, lat, hr_preds_samples[s], cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    im = axs[j][i, num_samples+1].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the row with the correct label
                    cbar = plt.colorbar(mappable=im, ax=axs[j][i, :num_samples+2], shrink=0.8, extend="max")
                    cbar.set_label(self.variables[i] + unit, fontsize=14)

                    # Plotting error sample sperately because of its different color scale
                    im_error = axs[j][i, num_samples+2].pcolormesh(lon, lat, error_sample, cmap=cmaps["error"], vmin=err_vmin, vmax=err_vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the error
                    cbar_error = plt.colorbar(mappable=im_error, ax=axs[j][i, num_samples+2], shrink=0.8, extend="max")
                    cbar_error.set_label(self.variables[i] + unit, fontsize=14)

                else:

                    cmap = cmaps["temp"]
                    unit = " (°C)"

                    # Converting units in °C and computing scaling values for diverging colormap
                    if self.variables[i] == "tasmin":
                        lr_sample, hr_preds_samples, hr_sample = KToC(lrinterp[j,i]), [KToC(hr_preds[j,s,i] for s in range(num_samples))], KToC(hr[j,i])
                    elif self.variables[i] == "tasmax":
                        if self.transfo:
                            lr_sample = KToC(softplus(lrinterp[j,i], c=0.) + lrinterp[j,i-1])
                            hr_preds_samples = [KToC(softplus(hr_preds[j,s,i], c=0.) + hr_preds[j,s,i-1]) for s in range(num_samples)]
                            hr_sample = KToC(softplus(hr[j,i], c=0.) + hr[j,i-1])
                        else:
                            lr_sample = KToC(lrinterp[j,i])
                            hr_preds_samples = [KToC(hr_preds[j,s,i]) for s in range(num_samples)]
                            hr_sample = KToC(hr[j,i])
                    max_abs = max(torch.amax(torch.abs(lr_sample)), torch.amax(torch.amax(hr_preds_samples)), torch.amax(torch.amax(hr_sample)))
                    vmin, vmax = -max_abs, max_abs

                    # Storing max_abs for computing shared vmin and vmax values for tasmin and tasmax later
                    temp_max_abs.append(max_abs)

                    # Computing absolute error and setting corresponding vmin, vmax
                    error_sample = torch.abs(hr_sample - torch.mean(torch.stack(hr_preds_samples), dim=0))
                    err_vmin, err_vmax = 0, torch.amax(error_sample)

                    # Setting cartopy features on the Axes objects
                    for l in range(total_cols):
                        axs[j][i, l].coastlines()
                        gl = axs[j][i, l].gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                        gl.top_labels = False
                        gl.right_labels = False
                        if l > 0:
                            gl.left_labels = False

                    # Plotting samples in the following order: upsampled low-resolution, predicted high-resolution, groundtruth high-resolution, error
                    axs[j][i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    for s in range(num_samples):
                        axs[j][i, s+1].pcolormesh(lon, lat, hr_preds_samples[s], cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    im = axs[j][i, num_samples+1].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the row with the correct label
                    cbar = plt.colorbar(mappable=im, ax=axs[j][i, :num_samples+2], shrink=0.8, extend="max")
                    cbar.set_label(self.variables[i] + unit, fontsize=14)

                    # Plotting error sample sperately because of its different color scale
                    im_error = axs[j][i, num_samples+2].pcolormesh(lon, lat, error_sample, cmap=cmaps["error"], vmin=err_vmin, vmax=err_vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the error
                    cbar_error = plt.colorbar(mappable=im_error, ax=axs[j][i, num_samples+2], shrink=0.8, extend="max")
                    cbar_error.set_label(self.variables[i] + unit, fontsize=14)

            shared_max_abs = np.max(temp_max_abs)
            for im in temp_ims:
                for im_c in im:
                    im_c.set_clim(vmin=-shared_max_abs, vmax=shared_max_abs)

            subfigs[j].suptitle(str(float_to_date(timestamps[j]))[:10], fontsize=16)

            axs[j][0, 0].set_title("Low-resolution", fontsize=14)
            for s in range(num_samples):
                axs[j][0, s+1].set_title("Prediction " + str(s+1), fontsize=14)
            axs[j][0, -2].set_title("High-resolution", fontsize=14)
            axs[j][0, -1].set_title("Absolute error", fontsize=14)

        fig.suptitle("Predictions after the " + str(epoch) + "th epoch for " + str(N) + " random validation dates", fontsize=18, fontweight='bold')

        return fig, axs