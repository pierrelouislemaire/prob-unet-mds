import torch
import numpy as np
from scipy import stats
import pysteps as ps

import codebase.data_utils as du
import codebase.train_utils as tu

def crps_over_groundtruth(hr, preds):

    # preds shape: (num_timestamps, num_samples, 3, h, w)
    # hr shape: (num_timestamps, 3, h, w)

    preds = preds.numpy()
    hr = hr.numpy()

    crps_vals = {var: [] for var in ["pr", "tasmin", "tasmax"]}
    for i in range(hr.shape[0]):
        hr_sample = hr[i].squeeze()
        preds_sample = preds[i].squeeze()
        crps_vals["pr"].append(ps.verification.probscores.CRPS(preds_sample[:, 0], hr_sample[0]))
        crps_vals["tasmin"].append(ps.verification.probscores.CRPS(preds_sample[:, 1], hr_sample[1]))
        crps_vals["tasmax"].append(ps.verification.probscores.CRPS(preds_sample[:, 2], hr_sample[2]))

    crps_vals["pr"] = np.mean(np.array(crps_vals["pr"]), axis=0)
    crps_vals["tasmin"] = np.mean(np.array(crps_vals["tasmin"]), axis=0)
    crps_vals["tasmax"] = np.mean(np.array(crps_vals["tasmax"]), axis=0)

    return crps_vals

def quantile_mae_rmse(ref, preds_list, list_names):

    num_variables = ref.shape[1]
    q_mae = {name: [] for name in list_names}
    q_rmse = {name: [] for name in list_names}

    for i in range(num_variables):
        bins = np.quantile(ref[:, i], np.arange(0, 1., 0.01))
        bins_idx = np.digitize(ref[:, i].flatten(), bins)
        quantile_binned_ref = [ref[:, i].flatten()[bins_idx == j] for j in range(1, len(bins)+1)]

        for preds, name in zip(preds_list, list_names):
            if len(preds.shape) == 5:
                preds = preds.mean(dim=1)
            quantile_binned_preds = [preds[:, i].flatten()[bins_idx == j] for j in range(1, len(bins)+1)]
            q_mae[name].append([np.abs(qb - qr).mean() for qb, qr in zip(quantile_binned_preds, quantile_binned_ref)])
            q_rmse[name].append([np.sqrt(((qb - qr)**2).mean()) for qb, qr in zip(quantile_binned_preds, quantile_binned_ref)])

    return q_mae, q_rmse

def psd(image):
    """
    Function to calculate the power spectral density of an image.
    """

    h, w = image.shape

    fourier_image = torch.fft.fftn(image)
    fourier_amplitudes = torch.abs(fourier_image)**2
    fourier_freq = torch.fft.fftfreq(h) * h 
    fourier_freq2d = np.meshgrid(fourier_freq, fourier_freq)
    fourier_freq2d_norm = np.sqrt(fourier_freq2d[0]**2 + fourier_freq2d[1]**2)

    fourier_amplitudes_f = fourier_amplitudes.flatten()
    fourier_freq2d_norm_f = fourier_freq2d_norm.flatten()

    kbins = np.arange(0.5, h//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    psd, _, _ = stats.binned_statistic(fourier_freq2d_norm_f, fourier_amplitudes_f,
                                        statistic = "mean",
                                        bins = kbins)
    psd *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals, psd

def compute_psd_over_groundtruth(hr, transfo):

    """
    Function to compute the power spectral density of the groundtruth (torch tensor).
    """

    hr_copy = hr.clone()
    if len(hr_copy.shape) == 5:
        hr_copy = hr_copy.mean(dim=1)

    psd_pr = []
    psd_tasmin = []
    psd_tasmax = []

    for sample in hr_copy:
        sample = sample.squeeze()
        if transfo:
            sample[0] = du.softplus(sample[0])
            sample[2] = du.softplus(sample[2], c=0) + sample[1]
        sample[0] = du.kgm2sTommday(sample[0])
        sample[1] = du.KToC(sample[1])
        sample[2] = du.KToC(sample[2])
        sample_pr = sample[0]
        sample_tasmin = sample[1]
        sample_tasmax = sample[2]
        _, psdvals_pr = psd(sample_pr)
        _, psdvals_tasmin = psd(sample_tasmin)
        _, psdvals_tasmax = psd(sample_tasmax)
        psd_pr.append(psdvals_pr)
        psd_tasmin.append(psdvals_tasmin)
        psd_tasmax.append(psdvals_tasmax)

    psd_pr = np.mean(np.array(psd_pr), axis=0)
    psd_tasmin = np.mean(np.array(psd_tasmin), axis=0)
    psd_tasmax = np.mean(np.array(psd_tasmax), axis=0)

    return psd_pr, psd_tasmin, psd_tasmax

def spread_skill_ratio(ref, ens_preds, reduction_dims=None):
    rmse = torch.sqrt(((ref - ens_preds.mean(dim=1)) ** 2).mean(dim=reduction_dims))
    var = torch.var(ens_preds, dim=1).mean(dim=reduction_dims)
    return rmse / torch.sqrt(var)

def sal(preds, ref):
    sal = []
    if len(preds.shape) == 5:
        preds = preds.mean(dim=1)
    for i in range(preds.shape[0]):
        sal.append(ps.verification.salscores.sal(preds[i, 0], ref[i, 0]))
    return np.nanmean(sal, axis=0)

