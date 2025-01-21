import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import properscoring as ps
import pysteps as pys

import climex_utils as cu

def crps(target, preds):
    h, w = target.shape[:1]
    crps = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            crps[i, j] = ps.crps(target[i, j], preds[:, i, j])
    return crps

def crps_over_groundtruth(hr, preds, transfo):

    # preds shape: (num_timestamps, num_samples, 3, h, w)
    # hr shape: (num_timestamps, 3, h, w)

    crps_vals = {var: [] for var in ["pr", "tasmin", "tasmax"]}
    for i in range(hr.shape[0]):
        hr_sample = hr[i].squeeze()
        preds_sample = preds[i].squeeze()
        crps_vals["pr"].append(crps(hr_sample[0], preds_sample[:, 0]))
        crps_vals["tasmin"].append(crps(hr_sample[1], preds_sample[:, 1]))
        crps_vals["tasmax"].append(crps(hr_sample[2], preds_sample[:, 2]))

    crps_vals["pr"] = np.mean(np.array(crps_vals["pr"]), axis=0)
    crps_vals["tasmin"] = np.mean(np.array(crps_vals["tasmin"]), axis=0)
    crps_vals["tasmax"] = np.mean(np.array(crps_vals["tasmax"]), axis=0)

    return crps_vals

def relative_bias(preds, targets, type="total"):

    preds = np.array(preds)
    targets = np.array(targets)

    if type=="total":
        num_preds = (preds - targets).mean(axis=(0, 2, 3))
        mean_targets = targets.mean(axis=(0, 2, 3))
        return num_preds / mean_targets * 100
    elif type=="spatial":
        num_preds = (preds - targets).mean(axis=(2, 3))
        mean_targets = targets.mean(axis=(2, 3))
        return num_preds / mean_targets * 100
    elif type=="temporal":
        num_preds = (preds - targets).mean(axis=0)
        mean_targets = targets.mean(axis=0)
        return num_preds / mean_targets * 100
    
def fss(preds, targets):
    fss_array = {"0.1": [], "1": [], "5": [], "15": []}
    for t in range([0.1, 1, 5, 15]):
        for r in range(preds.shape[-1]):
            fss = []
            for (pred, target) in zip(preds, targets):
                fss.append(pys.verification.spatialscores.fss(pred, target, r, t))
            fss_array[str(t)].append(np.mean(fss))
    return fss_array


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

    return kvals, psd, fourier_amplitudes

@torch.no_grad()
def compute_psd_over_loader(model, dataloader, device, transfo=False):

    """
    Function to compute the power spectral density of the predictions of a model over a dataloader.
    """
    
    model.eval()

    psd_pr = []
    psd_tasmin = []
    psd_tasmax = []

    fourier_amps_pr = []
    fourier_amps_tmin = []
    fourier_amps_tmax = []

    for i, batch in enumerate(dataloader):

        if dataloader.dataset.type == "lr_to_hr" or dataloader.dataset.type == "lrinterp_to_hr":
            inputs = batch['inputs'].to(device)
            preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
            hr_preds = dataloader.dataset.invstand_residual(preds.detach().cpu())
            if transfo:
                hr_preds[:, 0] = cu.softplus(hr_preds[:, 0])
                hr_preds[:, 2] = cu.softplus(hr_preds[:, 2], c=0) + hr_preds[:, 1]
            hr_preds[:, 0] = cu.kgm2sTommday(hr_preds[:, 0])
            hr_preds[:, 1] = cu.KToC(hr_preds[:, 1])
            hr_preds[:, 2] = cu.KToC(hr_preds[:, 2])
            for hr_pred in hr_preds:
                hr_pred = hr_pred.squeeze()
                hr_pred_pr = hr_pred[0]
                hr_pred_tasmin = hr_pred[1]
                hr_pred_tasmax = hr_pred[2]
                _, psdvals_pr, fourier_amp_pr = psd(hr_pred_pr)
                _, psdvals_tasmin, fourier_amp_tmin = psd(hr_pred_tasmin)
                _, psdvals_tasmax, fourier_amp_tmax = psd(hr_pred_tasmax)
                psd_pr.append(psdvals_pr)
                psd_tasmin.append(psdvals_tasmin)
                psd_tasmax.append(psdvals_tasmax)
                fourier_amps_pr.append(fourier_amp_pr)
                fourier_amps_tmin.append(fourier_amp_tmin)
                fourier_amps_tmax.append(fourier_amp_tmax)
                
        elif dataloader.dataset.type == "lrinterp_to_residuals" or dataloader.dataset.type == "lr_to_residuals":
            inputs, lrinterp = (batch['inputs'].to(device), batch['lrinterp'])
            preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
            hr_preds = dataloader.dataset.residual_to_hr(preds.detach().cpu(), lrinterp)
            if transfo:
                hr_preds[:, 0] = cu.softplus(hr_preds[:, 0])
                hr_preds[:, 2] = cu.softplus(hr_preds[:, 2], c=0) + hr_preds[:, 1]
            hr_preds[:, 0] = cu.kgm2sTommday(hr_preds[:, 0])
            hr_preds[:, 1] = cu.KToC(hr_preds[:, 1])
            hr_preds[:, 2] = cu.KToC(hr_preds[:, 2])
            for hr_pred in hr_preds:
                hr_pred = hr_pred.squeeze()
                hr_pred_pr = hr_pred[0]
                hr_pred_tasmin = hr_pred[1]
                hr_pred_tasmax = hr_pred[2]
                _, psdvals_pr, fourier_amp_pr = psd(hr_pred_pr)
                _, psdvals_tasmin, fourier_amp_tmin = psd(hr_pred_tasmin)
                _, psdvals_tasmax, fourier_amp_tmax = psd(hr_pred_tasmax)
                psd_pr.append(psdvals_pr)
                psd_tasmin.append(psdvals_tasmin)
                psd_tasmax.append(psdvals_tasmax)
                fourier_amps_pr.append(fourier_amp_pr)
                fourier_amps_tmin.append(fourier_amp_tmin)
                fourier_amps_tmax.append(fourier_amp_tmax)

        psd_pr = np.mean(np.array(psd_pr), axis=0)
        psd_tasmin = np.mean(np.array(psd_tasmin), axis=0)
        psd_tasmax = np.mean(np.array(psd_tasmax), axis=0)

        fourier_amps_pr = np.mean(np.array(fourier_amps_pr), axis=0)
        fourier_amps_tmin = np.mean(np.array(fourier_amps_tmin), axis=0)
        fourier_amps_tmax = np.mean(np.array(fourier_amps_tmax), axis=0)

        return psd_pr, psd_tasmin, psd_tasmax, fourier_amps_pr, fourier_amps_tmin, fourier_amps_tmax
    
def compute_psd_over_groundtruth(hr, transfo):

    """
    Function to compute the power spectral density of the groundtruth (torch tensor).
    """

    hr_copy = hr.clone()

    psd_pr = []
    psd_tasmin = []
    psd_tasmax = []

    fourier_amps_pr = []
    fourier_amps_tmin = []
    fourier_amps_tmax = []

    for sample in hr_copy:
        sample = sample.squeeze()
        if transfo:
            sample[0] = cu.softplus(sample[0])
            sample[2] = cu.softplus(sample[2], c=0) + sample[1]
        sample[0] = cu.kgm2sTommday(sample[0])
        sample[1] = cu.KToC(sample[1])
        sample[2] = cu.KToC(sample[2])
        sample_pr = sample[0]
        sample_tasmin = sample[1]
        sample_tasmax = sample[2]
        _, psdvals_pr, fourier_amp_pr = psd(sample_pr)
        _, psdvals_tasmin, fourier_amp_tmin = psd(sample_tasmin)
        _, psdvals_tasmax, fourier_amp_tmax = psd(sample_tasmax)
        psd_pr.append(psdvals_pr)
        psd_tasmin.append(psdvals_tasmin)
        psd_tasmax.append(psdvals_tasmax)
        fourier_amps_pr.append(fourier_amp_pr)
        fourier_amps_tmin.append(fourier_amp_tmin)
        fourier_amps_tmax.append(fourier_amp_tmax)

    psd_pr = np.mean(np.array(psd_pr), axis=0)
    psd_tasmin = np.mean(np.array(psd_tasmin), axis=0)
    psd_tasmax = np.mean(np.array(psd_tasmax), axis=0)

    fourier_amps_pr = np.mean(np.array(fourier_amps_pr), axis=0)
    fourier_amps_tmin = np.mean(np.array(fourier_amps_tmin), axis=0)
    fourier_amps_tmax = np.mean(np.array(fourier_amps_tmax), axis=0)

    return psd_pr, psd_tasmin, psd_tasmax, fourier_amps_pr, fourier_amps_tmin, fourier_amps_tmax

def compute_psd_over_groundtruth_prob(hr, transfo):

    """
    Function to compute the power spectral density of the groundtruth (torch tensor).
    """

    hr_copy = hr.clone()

    psd_pr = []
    psd_tasmin = []
    psd_tasmax = []

    fourier_amps_pr = []
    fourier_amps_tmin = []
    fourier_amps_tmax = []

    for sample in hr_copy:
        for gen_sample in sample:
            gen_sample = gen_sample.squeeze()
            if transfo:
                gen_sample[0] = cu.softplus(gen_sample[0])
                gen_sample[2] = cu.softplus(gen_sample[2], c=0) + gen_sample[1]
            gen_sample[0] = cu.kgm2sTommday(gen_sample[0])
            gen_sample[1] = cu.KToC(gen_sample[1])
            gen_sample[2] = cu.KToC(gen_sample[2])
            gen_sample_pr = gen_sample[0]
            gen_sample_tasmin = gen_sample[1]
            gen_sample_tasmax = gen_sample[2]
            _, psdvals_pr, fourier_amp_pr = psd(gen_sample_pr)
            _, psdvals_tasmin, fourier_amp_tmin = psd(gen_sample_tasmin)
            _, psdvals_tasmax, fourier_amp_tmax = psd(gen_sample_tasmax)
            psd_pr.append(psdvals_pr)
            psd_tasmin.append(psdvals_tasmin)
            psd_tasmax.append(psdvals_tasmax)
            fourier_amps_pr.append(fourier_amp_pr)
            fourier_amps_tmin.append(fourier_amp_tmin)
            fourier_amps_tmax.append(fourier_amp_tmax)

        psd_pr = np.mean(np.array(psd_pr), axis=0)
        psd_tasmin = np.mean(np.array(psd_tasmin), axis=0)
        psd_tasmax = np.mean(np.array(psd_tasmax), axis=0)

        fourier_amps_pr = np.mean(np.array(fourier_amps_pr), axis=0)
        fourier_amps_tmin = np.mean(np.array(fourier_amps_tmin), axis=0)
        fourier_amps_tmax = np.mean(np.array(fourier_amps_tmax), axis=0)

        return psd_pr, psd_tasmin, psd_tasmax, fourier_amps_pr, fourier_amps_tmin, fourier_amps_tmax
