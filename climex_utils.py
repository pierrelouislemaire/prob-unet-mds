import glob
import xarray as xr
import numpy as np
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
        
# For temperature
def KToC(data):
    return data - 273.15

####

class climexSet(Dataset):

    """
    Dataset class that loads and converts data from NetCDF files to Pytorch tensors on initialization. 
    climexSet object can be fed to a Pytorch Dataloader.
    """

    def __init__(self, datadir, years=range(1960,2039), variables=["pr", "tasmin", "tasmax"], coords=[150, 182, 135, 167], train = True, trainset = None, lowres_scale = 4, time_transform=None):

        """
        datadir: (str) path to the directory containing NetCDF files;
        years: (list of int) indicates which years should climexSet import data from;
        variables: (list of str) indicates what variables should climexSet import data from;
        coords: (list of int) (form: [start_rlon, end_rlon, start_rlat, end_rlat]) climexSet will only import data from the resulting window:
        train: (bool) True if used for training data (important for standardization), False otherwise;
        trainset: (climexSet) used if train==False, should be the corresponding training dataset (important for standardization);
        lowres_scale: (int) downscaling factor.
        """

        super().__init__()

        self.variables = variables
        self.nvars = len(variables)
        self.train = train
        self.trainset = trainset
        self.lowres_scale = lowres_scale
        self.time_transform = time_transform
        self.epsilon = 1e-10 #used for standardization

        if train:
            print("Generating train set")
        else:
            print("Generating test set")

        # Recursively getting all NetCDF files names
        files = []
        for year in years:
            for var in variables:
                files.append(glob.glob("{path}/*_{var}_*_{year}_*".format(path=datadir, var=var, year=year))[0])    

        # Importing all NetCDF files into a xarray Dataset with lazy loading and chunking
        self.data = xr.open_mfdataset(paths=files, chunks={"time":100, "rlon": 100, "rlat":100}, data_vars="minimal", coords="minimal", compat="override")[self.variables]

        # Extracting latitude and longitude data (for plotting function)
        self.lon = self.data.isel(rlon=range(coords[0], coords[1]), rlat=range(coords[2],coords[3])).lon
        self.lat = self.data.isel(rlon=range(coords[0], coords[1]), rlat=range(coords[2],coords[3])).lat

        print("Imported all data files into xarray Dataset using lazy loading")

        # Loading into memory high-resolution ground-truth data from desired spatial window and converting to Pytorch tensor (time, nvar, height, width)
        self.hr = torch.tensor(self.data.isel(rlon=slice(coords[0], coords[1]), rlat=slice(coords[2],coords[3])).load().to_array().to_numpy()).transpose(0,1)

        print("Loaded dataset into memory")

        # Averging high-resolution ground-truth to generate low-resolution 
        self.lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(self.hr)

        # Interpolating back low-resolution to high-resolution using bilinear upsampling (this will be used as the inputs)
        self.lrinterp = nn.UpsamplingBilinear2d(scale_factor=self.lowres_scale)(self.lr)

        # Computing the residual between the ground-truth and the upsampled approximation
        self.residual = self.hr - self.lrinterp

        if train:
            # Computing training statistics for standardization
            self.mean_lrinterp, self.std_lrinterp = self.lrinterp.mean(dim=0, keepdim=True), self.lrinterp.std(dim=0, keepdim=True)
            self.mean_residual, self.std_residual = self.residual.mean(dim=0, keepdim=True), self.residual.std(dim=0, keepdim=True)
        else:
            # Recovering training statistics for standardization (avoids data leak)
            self.mean_lrinterp, self.std_lrinterp = self.trainset.mean_lrinterp, self.trainset.std_lrinterp
            self.mean_residual, self.std_residual = self.trainset.mean_residual, self.trainset.std_residual
        
        # Standardizing upsampled and residual data to be used by the model
        self.inputs = (self.lrinterp - self.mean_lrinterp)/(self.std_lrinterp + self.epsilon)
        self.targets = (self.residual - self.mean_residual)/(self.std_residual + self.epsilon)

        # Extracting the timestamps to inform the model about seasonal cycles and long-range trends
        self.timestamps = torch.tensor(date_to_float(self.data.indexes["time"].to_datetimeindex()))

        if self.time_transform:
            self.timestamps = self.time_transform(self.timestamps)

        print("")
        print("##########################################")
        print("############ PROCESSING DONE #############")
        print("##########################################")
        print("")


    def __len__(self):
         return len(self.timestamps)

    def __getitem__(self, idx):

        return {"inputs": self.inputs[idx],
                "targets": self.targets[idx],
                "timestamps": self.timestamps[idx],
                "hr": self.hr[idx], 
                "lr": self.lr[idx],
                "lrinterp": self.lrinterp[idx]}

    # Computes the inverse of the standardization for the residual
    def invstand_residual(self, standardized_residual):
        return standardized_residual * (self.std_residual + self.epsilon) + self.mean_residual
    
    # Adds the predicted residual to the input upsampled high-resolution
    def residual_to_hr(self, residual, lrinterp):
        return lrinterp + self.invstand_residual(residual)
    
    # Plot a batch (N) of samples (upsampled low-resolution, predicted high-resolution, groundtruth high-resolution)
    def plot_batch(self, lrinterp, hr_pred, hr, timestamps, epoch, N=2):

        # Initializing Plate Carrée projection (for other projections see https://scitools.org.uk/cartopy/docs/latest/reference/crs.html)
        prj = ccrs.PlateCarree()

        # Initializing figure and subfigures (one subfigure per date)
        fig = plt.figure(figsize=(N * 18, 12), constrained_layout=True)
        subfigs = fig.subfigures(1, N, wspace=0.05)

        # Different colormaps for different type of climate variables
        cmaps = {'pr': cm.get_cmap('GnBu'), 'temp': cm.get_cmap('coolwarm')}

        axs = []
        # Batch (N) plotting loop
        for j in range(N):

            axs.append(subfigs[j].subplots(self.nvars, 3, subplot_kw={'projection': prj}, gridspec_kw={'wspace':0.01, 'hspace':0.005}))

            # Extracting latitude and longitude data corresponding to the j-th sample from the batch
            lat, lon = self.lat.sel(time=str(float_to_date(timestamps[j]))[:10]).load().to_numpy().squeeze(), self.lon.sel(time=str(float_to_date(timestamps[j]))[:10]).load().to_numpy().squeeze()
            lonE, lonW = lon.min(), lon.max()
            latS, latN = lat.min(), lat.max()

            # Variables plotting loop
            for i in range(self.nvars):

                if self.variables[i] == "pr":

                    cmap = cmaps["pr"]
                    unit = " (mm/day)"

                    # Converting units in mm/day and computing scaling values for colormap
                    lr_sample, hr_pred_sample, hr_sample = kgm2sTommday(lrinterp[j,i]), kgm2sTommday(hr_pred[j,i]), kgm2sTommday(hr[j,i])
                    vmin, vmax = 0, max(torch.amax(lr_sample), torch.amax(hr_pred_sample), torch.amax(hr_sample))

                    # Setting cartopy features on the Axes objects
                    for l in range(3):
                        axs[j][i, l].set_extent([lonW, lonE, latS, latN], crs=prj)
                        axs[j][i, l].coastlines()
                        gl = axs[j][i, l].gridlines(crs=prj, draw_labels=True)
                        gl.xlines = False
                        gl.ylines = False
                        gl.top_labels = False
                        gl.right_labels = False
                        if l > 0:
                            gl.left_labels = False

                    # Plotting samlpes in the following order: upsampled low-resolution, predicted high-resolution, groundtruth high-resolution
                    axs[j][i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=prj)
                    axs[j][i, 1].pcolormesh(lon, lat, hr_pred_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=prj)
                    im = axs[j][i, 2].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=prj)

                    # Plotting the colorbar for the row with the correct label
                    cbar = plt.colorbar(mappable=im, ax=axs[j][i, :], shrink=0.8, extend="max")
                    cbar.set_label(self.variables[i] + unit, fontsize=14)

                else:

                    cmap = cmaps["temp"]
                    unit = " (°C)"

                    # Converting units in °C and computing scaling values for diverging colormap
                    lr_sample, hr_pred_sample, hr_sample = KToC(lrinterp[j,i]), KToC(hr_pred[j,i]), KToC(hr[j,i])
                    max_abs =  max(torch.amax(torch.abs(lr_sample)), torch.amax(torch.amax(hr_pred_sample)), torch.amax(torch.amax(hr_sample)))
                    vmin, vmax = -max_abs, max_abs

                    # Setting cartopy features on the Axes objects
                    for l in range(3):
                        axs[j][i, l].set_extent([lonW, lonE, latS, latN], crs=prj)
                        axs[j][i, l].coastlines()
                        gl = axs[j][i, l].gridlines(crs=prj, draw_labels=True)
                        gl.xlines = False
                        gl.ylines = False
                        gl.top_labels = False
                        gl.right_labels = False
                        if l > 0:
                            gl.left_labels = False

                    # Plotting samlpes in the following order: upsampled low-resolution, predicted high-resolution, groundtruth high-resolution
                    axs[j][i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=prj)
                    axs[j][i, 1].pcolormesh(lon, lat, hr_pred_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=prj)
                    im = axs[j][i, 2].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=prj)

                    # Plotting the colorbar for the row with the correct label
                    cbar = plt.colorbar(mappable=im, ax=axs[j][i, :], shrink=0.8, extend="both")
                    cbar.set_label(self.variables[i] + unit, fontsize=14)


            subfigs[j].suptitle(str(float_to_date(timestamps[j]))[:10], fontsize=16)

        axs[0][0, 0].set_title("Low-resolution", fontsize=14)
        axs[0][0, 1].set_title("Prediction", fontsize=14)
        axs[0][0, 2].set_title("High-resolution", fontsize=14)

        axs[1][0, 0].set_title("Low-resolution", fontsize=14)
        axs[1][0, 1].set_title("Prediction", fontsize=14)
        axs[1][0, 2].set_title("High-resolution", fontsize=14)

        fig.suptitle("Predictions after the " + str(epoch) + "th epoch for 2 random test dates", fontsize=18, fontweight='bold')

        plt.show()

        return fig, axs
    


