import torch
import xarray as xr

def BCSD(datatrain, datatest):

    ds_factor = datatrain.lowres_scale

    lr_proj = datatest.data.to_array().coarsen(rlat=ds_factor, rlon=ds_factor).mean().groupby('time.month')
    lr_monthly_avg = datatrain.data.to_array().coarsen(rlat=ds_factor, rlon=ds_factor).mean().groupby('time.month').mean()
    scale = lr_proj - lr_monthly_avg

    scale_interp_data = torch.nn.functional.interpolate(torch.from_numpy(scale.data), scale_factor=ds_factor, mode='bilinear')
    scale_interp = xr.DataArray(scale_interp_data, coords={"time": scale.time, "rlat": datatrain.data.rlat, "rlon":datatrain.data.rlon}, dims=scale.dims)
    bcsd = datatrain.data.to_array().groupby('time.month').mean() + scale_interp.groupby('time.month')

    return torch.from_numpy(bcsd.to_numpy()).transpose(0, 1)