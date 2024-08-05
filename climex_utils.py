import glob
import dask
from dask.distributed import Client
import xarray as xr
import numpy as np
import bottleneck

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
        
# For temperature
def KToC(data):
    return data - 273.15

####

class climex2torch(Dataset):

    """
    Dataset class that loads and converts data from NetCDF files to Pytorch tensors on initialization. 
    climex2torch object can be fed to a Pytorch Dataloader.
    """

    def __init__(self, datadir, years=range(1960,2039), variables=["pr", "tasmin", "tasmax"], coords=[150, 182, 135, 167], train = True, trainset = None, lowres_scale = 4, time_transform=None):

        """
        datadir: (str) path to the directory containing NetCDF files;
        years: (list of int) indicates which years should climex2torch import data from;
        variables: (list of str) indicates what variables should climex2torch import data from;
        coords: (list of int) (form: [start_rlon, end_rlon, start_rlat, end_rlat]) climex2torch will only import data from the resulting window:
        train: (bool) True if used for training data (important for standardization), False otherwise;
        trainset: (climex2torch) used if train==False, should be the corresponding training dataset (important for standardization);
        lowres_scale: (int) downscaling factor.
        """

        super().__init__()

        # Setup dask distributed cluster
        client = Client()

        self.variables = variables
        self.nvars = len(variables)
        self.coords = coords
        self.width = self.coords[1] - self.coords[0]
        self.height = self.coords[3] - self.coords[2]
        self.train = train
        self.trainset = trainset
        self.lowres_scale = lowres_scale
        self.time_transform = time_transform
        self.epsilon = 1e-10 #used for standardization

        # Optimization of chunk sizes so chunks are around 100 mb
        self.chunksize = int(28616000/(self.height * self.width))
        if self.chunksize > 365 * len(years):
            self.chunksize = 365 * len(years)

        # Preprocessing function to select only desired coordinates
        def select_coords(ds):
            return ds.isel(rlon=slice(coords[0], coords[1]), rlat=slice(coords[2],coords[3]))

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
        self.data = xr.open_mfdataset(paths=files, engine='h5netcdf', preprocess=select_coords, data_vars="minimal", coords="minimal", compat="override", parallel=True)[self.variables].chunk({"time": self.chunksize, "rlon": self.width, "rlat": self.height})
        
        # Extracting latitude and longitude data (for plotting function)
        self.lon = self.data.lon
        self.lat = self.data.lat

        self.data = self.data.to_array()

        print("Imported all data files into xarray Dataset using lazy loading")

        # Loading into memory high-resolution ground-truth data from desired spatial window and converting to Pytorch tensor (time, nvar, height, width)
        self.data.load()
        self.hr = torch.from_numpy(self.data.data).transpose(0,1)

        print("Loaded dataset into memory")

        # Averging high-resolution ground-truth to generate low-resolution 
        self.lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(self.hr)

        # Interpolating back low-resolution to high-resolution using bilinear upsampling (this will be used as the inputs)
        self.lrinterp = nn.UpsamplingBilinear2d(scale_factor=self.lowres_scale)(self.lr)

        # Computing the residual between the ground-truth and the upsampled approximation (this will be used as the targets after being standardized)
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

        # Initializing Plate Carrée and Rotated Pole projections (for other projections see https://scitools.org.uk/cartopy/docs/latest/reference/crs.html)
        rotatedpole_prj = ccrs.RotatedPole(pole_longitude=83.0, pole_latitude=42.5)
        platecarree_proj = ccrs.PlateCarree()

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

            axs.append(subfigs[j].subplots(self.nvars, 4, subplot_kw={'projection': rotatedpole_prj}, gridspec_kw={'wspace':0.01, 'hspace':0.005}))

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
                    lr_sample, hr_pred_sample, hr_sample = kgm2sTommday(lrinterp[j,i]), kgm2sTommday(hr_pred[j,i]), kgm2sTommday(hr[j,i])
                    vmin, vmax = 0, max(torch.amax(lr_sample), torch.amax(hr_pred_sample), torch.amax(hr_sample))

                    # Computing absolute error and setting corresponding vmin, vmax
                    error_sample = torch.abs(hr_sample - hr_pred_sample)
                    err_vmin, err_vmax = 0, torch.amax(error_sample)

                    # Setting cartopy features on the Axes objects
                    for l in range(4):
                        axs[j][i, l].coastlines()
                        gl = axs[j][i, l].gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                        gl.top_labels = False
                        gl.right_labels = False
                        if l > 0:
                            gl.left_labels = False

                    # Plotting samples in the following order: upsampled low-resolution, predicted high-resolution, groundtruth high-resolution
                    axs[j][i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    axs[j][i, 1].pcolormesh(lon, lat, hr_pred_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    im = axs[j][i, 2].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the row with the correct label
                    cbar = plt.colorbar(mappable=im, ax=axs[j][i, :3], shrink=0.8, extend="max")
                    cbar.set_label(self.variables[i] + unit, fontsize=14)

                    # Plotting error sample sperately because of its different color scale
                    im_error = axs[j][i, 3].pcolormesh(lon, lat, error_sample, cmap=cmaps["error"], vmin=err_vmin, vmax=err_vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the error
                    cbar_error = plt.colorbar(mappable=im_error, ax=axs[j][i, 3], shrink=0.8, extend="max")
                    cbar_error.set_label(self.variables[i] + unit, fontsize=14)

                else:

                    cmap = cmaps["temp"]
                    unit = " (°C)"

                    # Converting units in °C and computing scaling values for diverging colormap
                    lr_sample, hr_pred_sample, hr_sample = KToC(lrinterp[j,i]), KToC(hr_pred[j,i]), KToC(hr[j,i])
                    max_abs = max(torch.amax(torch.abs(lr_sample)), torch.amax(torch.amax(hr_pred_sample)), torch.amax(torch.amax(hr_sample)))
                    vmin, vmax = -max_abs, max_abs

                    # Storing max_abs for computing shared vmin and vmax values for tasmin and tasmax later
                    temp_max_abs.append(max_abs)

                    # Computing absolute error and setting corresponding vmin, vmax
                    error_sample = torch.abs(hr_sample - hr_pred_sample)
                    err_vmin, err_vmax = 0, torch.amax(error_sample)

                    # Setting cartopy features on the Axes objects
                    for l in range(4):
                        axs[j][i, l].coastlines()
                        gl = axs[j][i, l].gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                        gl.top_labels = False
                        gl.right_labels = False
                        if l > 0:
                            gl.left_labels = False

                    # Plotting samlpes in the following order: upsampled low-resolution, predicted high-resolution, groundtruth high-resolution
                    axs[j][i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    axs[j][i, 1].pcolormesh(lon, lat, hr_pred_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    im = axs[j][i, 2].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the row with the correct label
                    cbar = plt.colorbar(mappable=im, ax=axs[j][i, :3], shrink=0.8, extend="both")
                    cbar.set_label(self.variables[i] + unit, fontsize=14)

                    temp_ims.append(im)

                    # Plotting error sample sperately because of its different color scale
                    im_error = axs[j][i, 3].pcolormesh(lon, lat, error_sample, cmap=cmaps["error"], vmin=err_vmin, vmax=err_vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the error
                    cbar_error = plt.colorbar(mappable=im_error, ax=axs[j][i, 3], shrink=0.8, extend="max")
                    cbar_error.set_label(self.variables[i] + unit, fontsize=14)

            shared_max_abs = np.max(temp_max_abs)
            for im in temp_ims:
                im.set_clim(vmin=-shared_max_abs, vmax=shared_max_abs)

            subfigs[j].suptitle(str(float_to_date(timestamps[j]))[:10], fontsize=16)

            axs[j][0, 0].set_title("Low-resolution", fontsize=14)
            axs[j][0, 1].set_title("Prediction", fontsize=14)
            axs[j][0, 2].set_title("High-resolution", fontsize=14)
            axs[j][0, 3].set_title("Absolute error", fontsize=14)

        fig.suptitle("Predictions after the " + str(epoch) + "th epoch for " + str(N) + " random test dates", fontsize=18, fontweight='bold')

        plt.show()

        return fig, axs
    

#######

class climexEDA:

    """
    climexEDA lazy loads the dataset and perform numerous statistical computations, including plotting, without loading the whole dataset into memory at once using xarray and dask.
    """

    def __init__(self, datadir, years=range(1960, 2099), variables=["pr", "tasmin", "tasmax"], coords=[0, 280, 0, 280]):

        """
        datadir: (str) path to the directory containing NetCDF files;
        years: (list of int) indicates which years should climexEDA import data from;
        variables: (list of str) indicates what variables should climexEDA import data from;
        coords: (list of int) (form: [start_rlon, end_rlon, start_rlat, end_rlat]) climexEDA will only import data from the resulting window:
        """

        self.datadir = datadir
        self.years = years
        self.variables = variables
        self.nvars = len(self.variables)
        self.coords = coords
        self.width = self.coords[1] - self.coords[0]
        self.height = self.coords[3] - self.coords[2]

        # Setup dask scheduler (not distributed here because of Spearman computation excessive memory usage)
        dask.config.set(scheduler="threads")

        # Optimization of chunk sizes so chunks are around 100 mb
        self.chunksize = int(28616000/(self.height * self.width))
        if self.chunksize > 365 * len(years):
            self.chunksize = 365 * len(years)

        # Preprocessing function to select only desired coordinates
        def select_coords(ds):
            return ds.isel(rlon=slice(self.coords[0], self.coords[1]), rlat=slice(self.coords[2], self.coords[3]))

        # Recursively getting all NetCDF files names
        files = []
        for year in years:
            for var in variables:
                files.append(glob.glob("{path}/*_{var}_*_{year}_*".format(path=datadir, var=var, year=year))[0])    

        # Importing all NetCDF files into a xarray Dataset with lazy loading and chunking
        self.data = xr.open_mfdataset(paths=files, engine='h5netcdf', preprocess=select_coords, data_vars="minimal", coords="minimal", compat="override")[self.variables].chunk({"time": self.chunksize, "rlon": self.width, "rlat": self.height})

        # Extracting latitude and longitude data (for plotting function)
        self.lon = self.data.lon.isel(time=0).load().to_numpy().squeeze()
        self.lat = self.data.lat.isel(time=0).load().to_numpy().squeeze()

        # Group data by seasons for later use
        self.season_groupby = self.data.groupby('time.season')
        self.seasons = np.array(["DJF", "MAM", "JJA", "SON"])

        # Plate Carrée and Rotated Pole projections (for other projections see https://scitools.org.uk/cartopy/docs/latest/reference/crs.html)
        self.rotatedpole_prj = ccrs.RotatedPole(pole_longitude=83.0, pole_latitude=42.5)
        self.platecarree_proj = ccrs.PlateCarree()
        
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
        self.cmaps = {'pr': prep_colormap, 'temp': cm.get_cmap('RdBu_r'), 'corr': cm.get_cmap('gist_rainbow')}

    # Computes the desired interannual seasonsal statistics pixel per pixel 
    def interannual_stat_ppp_seasonal(self, stat="mean", load=False, plot=False):

        # Supported statistics are mean, median, 1st and 3rd quartile, min and max

        if stat == "mean":
            interannual_stat_seasonal = self.season_groupby.mean(dim="time")
        elif stat == "median":
            interannual_stat_seasonal = self.season_groupby.median(dim="time")
        # needs to unchunk the data on the time dimension for quartiles computation because xarray apply a numpy function using Dask (w/ apply_ufunc)
        elif stat == "1st-quartile":
            interannual_stat_seasonal = self.data.chunk({'time':-1}).groupby("time.season").quantile(q=0.25, dim="time")
        elif stat == "3rd-quartile":
            interannual_stat_seasonal = self.data.chunk({'time':-1}).groupby("time.season").quantile(q=0.75, dim="time")
        elif stat == "min":
            interannual_stat_seasonal = self.season_groupby.min(dim="time")
        elif stat == "max":
            interannual_stat_seasonal = self.season_groupby.max(dim="time")
        else:
            raise ValueError("Received unknown statistics")

        # Loads the results into memory and actually performs the computation
        if load:
            interannual_stat_seasonal.load()

        # Plot the results and return the plots as (fig, axs)
        if plot:
            return interannual_stat_seasonal, self.plot_grids_seasonal(interannual_stat_seasonal, "Interannual seasonal " + stat)
        else:
            return interannual_stat_seasonal
            
    # For each variable, computes a profile over one dim for each day of year
    def annual_cycle_along_dim(self, dim="rlat", load=True, plot=False):

        # dim is the profile dimension, so we want to average over the other one
        if dim == "rlat":
            avg_dim = "rlon"
        elif dim == "rlon":
            avg_dim = "rlat"
        else:
            raise ValueError("unrecognized dimension")
        
        # Groups by and averaging
        annual_cycle_along_dim = self.data.groupby("time.dayofyear").mean(dim=[avg_dim, "time"])

        # Loads the results into memory and actually performs the computation 
        if load:
            annual_cycle_along_dim.load()

        # Plots the results and returns the plotting objects
        if plot:

            # rlon or rlat values
            dim_values = self.data[dim].load()

            fig, axs = plt.subplots(1, self.nvars, figsize=(15, 5), constrained_layout=True)
            for v in range(self.nvars):

                # Loads the correct DataArray into memory before to avoid redundant loading
                data2plot = annual_cycle_along_dim[self.variables[v]].load()

                if self.variables[v] == "pr":
                    unit = " (mm/day)"
                    data2plot = kgm2sTommday(data2plot)
                else:
                    unit = " (°C)"
                    data2plot = KToC(data2plot)

                # Initialize colormap and mappable for the plot legend over dim_values
                normalize = mpl.colors.Normalize(vmin=dim_values.min(), vmax=dim_values.max())
                cmap = cm.get_cmap('viridis')

                for dim_val in dim_values:
                    axs[v].plot(data2plot.sel({dim: dim_val}), color=cmap(normalize(dim_val)), lw=1)

                axs[v].set_ylabel(self.variables[v] + unit)
                axs[v].set_xlabel("day of year")

            # Display colorbar as continuous legend
            scalarmappable = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappable.set_array(dim_values)
            cbar = plt.colorbar(scalarmappable, ax=axs[-1])
            cbar.set_label(dim)

            fig.suptitle("Annual cycle over " + dim, fontsize=18)
            plt.show()

            return annual_cycle_along_dim, (fig, axs)
        
        else:

            return annual_cycle_along_dim

    # Helper function that computes the spearman correlation coefficient between two 1D arrays
    def spearman_gufunc(self, x, y):

        def covariance_gufunc(x, y):
            return ((x - x.mean(axis=-1, keepdims=True)) * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

        def pearson_correlation_gufunc(x, y):
            return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))
        
        # bottleneck is just a faster implementation of some numpy functions
        x_ranks = bottleneck.rankdata(x, axis=-1)
        y_ranks = bottleneck.rankdata(y, axis=-1)
        return pearson_correlation_gufunc(x_ranks, y_ranks)
    
    # Computes the spearman correlation for each pixel wrt a reference pixel, seasonal
    def spearmancorr_seasonal_fop(self, pxl_coords=[32, 32], load=True, plot=False):        

        # In case we didn't load the full domain, check coordinates and adjust to the new domain reference
        if pxl_coords[0] < self.coords[0] or pxl_coords[0] > self.coords[1] or pxl_coords[1] < self.coords[2] or pxl_coords[1] > self.coords[3]:
            raise ValueError("Rotated coordinates incorrect")
        else:
            pxl_rlat = pxl_coords[0] - self.coords[0]
            pxl_rlon = pxl_coords[1] - self.coords[2]

        # For optimization, because time will be unchunked, adjust other chunks so chunks are about 100 mb
        rechunk = int(np.sqrt(28616000/self.chunksize))

        spearman = {}
        # Loop over seasons
        for season in self.seasons:

            # Drop latitude and longitude because not necessary and REALLY slow the computation
            # Unchunk the time dimension to apply the spearman_gufunc
            season_data = self.season_groupby[season].drop_vars(["lat", "lon"]).chunk({'time':-1, 'rlat': rechunk, 'rlon': rechunk})

            # Apply a function that normally takes ndarray as input to a DataArray and perform correct broadcasting + enable dask parallel computation
            spearman[season] = xr.apply_ufunc(self.spearman_gufunc, season_data.isel(rlat=pxl_rlat, rlon=pxl_rlon), 
                                                season_data, input_core_dims=[["time"], ["time"]], output_core_dims=[[]], 
                                                dask="parallelized", output_dtypes=[np.float32])

        # Turn dict keys into a dimension in a xr.Dataset
        spearman = xr.concat([spearman[season] for season in self.seasons], dim=xr.Variable(dims="season", data=self.seasons))
        
        # Loads the results into memory and actually performs the computation 
        if load:
            # Need to loop over seasons because loading everything at once makes the kernel die
            for season in self.seasons:
                spearman.sel(season=season).load()

        # Plot the results and return the plots as (fig, axs)
        if plot:
            plot_title = "Spearman correlation for latitude = " + str(self.lat[pxl_rlat, pxl_rlon]) + "° and longitude = " + str(self.lon[pxl_rlat, pxl_rlon]) + "°"
            return spearman, self.plot_grids_seasonal(dataset=spearman, title=plot_title, correlation=True)
        else:
            return spearman
    
    # Compute autocorrelation using spearman with lags in number of days for a given pixel
    def autocorr_spearman_lag_ppp(self, pxl_coords=[140, 140], load=True, plot=False):

        # In case we didn't load the full domain, check coordinates and adjust to the new domain reference
        if pxl_coords[0] < self.coords[0] or pxl_coords[0] > self.coords[1] or pxl_coords[1] < self.coords[2] or pxl_coords[1] > self.coords[3]:
            raise ValueError("Rotated coordinates incorrect")
        else:
            pxl_rlat = pxl_coords[0] - self.coords[0]
            pxl_rlon = pxl_coords[1] - self.coords[2]

        # Extract data for the chosen pixel, unchunk time for apply_ufunc, and drop latitude and longitude
        pxl_data = self.data.drop_vars(["lat", "lon"]).isel(rlat=pxl_rlat, rlon=pxl_rlon).chunk({'time':-1})
        # Concatenate shifted versions of the array to create lags and group by seasons
        pxl_lags = xr.concat([pxl_data.shift(time=t) for t in range(1,31)], dim="lags").groupby("time.season")
        # Group the unlagged pixel data by season
        pxl_data = pxl_data.groupby("time.season")

        spearman = {}
        # Loop over seasons
        for season in self.seasons:
           
           # Apply a function that normally takes ndarray as input to a DataArray and perform correct broadcasting + enable dask parallel computation
           spearman[season] = xr.apply_ufunc(self.spearman_gufunc, pxl_data[season], pxl_lags[season], input_core_dims=[["time"], ["time"]], output_core_dims=[[]], dask="parallelized", output_dtypes=[np.float32])

        # Turn dict keys into a dimension in a xr.Dataset
        spearman = xr.concat([spearman[season] for season in self.seasons], dim=xr.Variable(dims="season", data=self.seasons))

        # Loads the results into memory and actually performs the computation 
        if load:
            # Need to loop over seasons because loading everything at once makes the kernel die
            for season in self.seasons:
                spearman.sel(season=season).load()

        # Plots the results and return the plotting objects (fig, axs)
        if plot:

            fig, axs = plt.subplots(self.nvars, len(self.seasons), figsize=(15, 10), constrained_layout=True)
            for i in range(self.nvars):
                for j in range(len(self.seasons)):

                    # Loads the correct data array to plot on the ax before to avoid redundant loading
                    data2plot = spearman[self.variables[i]].sel(season=self.seasons[j]).load()

                    axs[i,j].bar(x=range(1,31), height=data2plot, width=0.1, color='black')
                    axs[i,j].set_ylim(-.2, 1.)
                    axs[i,j].axhline(y = 0.05, color='blue', linestyle='--', lw=1)
                    axs[i,j].axhline(color='black', lw=1)
                    axs[i,j].axhline(y = -0.05, color='blue', linestyle='--', lw=1)
                    axs[i,j].set_xlabel("Lags in days")
                    if i == 0:
                        axs[i,j].set_title(self.variables[i])
                
                axs[i,0].set_ylabel("Spearman autocorrelation for " + self.variables[i])

            fig.suptitle("Spearman autocorrelation with lags for latitude = " + str(self.lat[pxl_rlat, pxl_rlon]) + "° and longitude = " + str(self.lon[pxl_rlat, pxl_rlon]) + "°", fontsize=18)
            plt.show()

            return spearman, (fig, axs)
        
        else:
            return spearman

    # Plotting function for pixel per pixel seasonal statistics
    def plot_grids_seasonal(self, dataset, title, correlation=False):

        fig, axs = plt.subplots(nrows=self.nvars, ncols=4, figsize=(15, 10), subplot_kw={'projection': self.rotatedpole_prj}, constrained_layout=True)
        for i in range(self.nvars):

            # Separate correlation statistics because units are not preserved and scale is different
            if correlation:
                cmap = self.cmaps["corr"]
                unit = ""
                extend = "neither"
                data2plot = dataset[self.variables[i]].load()
                vmin, vmax = data2plot.min(), 1
            else:
                if self.variables[i] == "pr":
                    cmap = self.cmaps["pr"]
                    unit = " (mm/day)"
                    extend = "max"
                    # Apply unit conversion over the loaded array
                    data2plot = xr.apply_ufunc(kgm2sTommday, dataset[self.variables[i]].load())
                    vmin, vmax = 0, data2plot.max()
                else:
                    cmap = self.cmaps["temp"]
                    unit = " (°C)"
                    extend = "both"
                    # Apply unit conversion over the loaded array
                    data2plot = xr.apply_ufunc(KToC, dataset[self.variables[i]].load())
                    max_abs = np.max([-data2plot.min(), data2plot.max()])
                    vmin, vmax = -max_abs, max_abs

            # Loop over seasons to plot
            for j in range(len(self.seasons)):
                if i == 0:
                    axs[i,j].set_title(self.seasons[j], fontsize=14)
                # Activate cartopy features
                axs[i,j].coastlines()
                gl = axs[i, j].gridlines(crs=self.platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                gl.top_labels = False
                gl.right_labels = False

                im = axs[i,j].pcolormesh(self.lon, self.lat, data2plot.sel(season=self.seasons[j]), cmap=cmap, vmin=vmin, vmax=vmax, transform=self.platecarree_proj)

            # Plotting the colorbar for the row with the correct label
            cbar = plt.colorbar(mappable=im, ax=axs[i,:], shrink=0.8, extend=extend)
            cbar.set_label(self.variables[i] + unit, fontsize=14)

        fig.suptitle(title, fontsize=18)

        plt.show()

        return (fig, axs)