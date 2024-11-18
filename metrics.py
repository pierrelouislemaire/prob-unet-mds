import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import climex_utils as cu

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

    fourier_amplitudes = fourier_amplitudes.flatten()
    fourier_freq2d_norm = fourier_freq2d_norm.flatten()

    kbins = np.arange(0.5, h//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    psd, _, _ = stats.binned_statistic(fourier_freq2d_norm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    psd *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals, psd

@torch.no_grad()
def compute_psd_over_loader(model, dataloader, device, transfo=False):

    """
    Function to compute the power spectral density of the predictions of a model over a dataloader.
    """
    
    model.eval()

    psd_pr = []
    psd_tasmin = []
    psd_tasmax = []

    for i, batch in enumerate(dataloader):

        if dataloader.dataset.type == "lr_to_hr" or dataloader.dataset.type == "lrinterp_to_hr":
            inputs = batch['inputs'].to(device)
            preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
            hr_preds = dataloader.dataset.invstand_residual(preds.detach().cpu())
            if transfo:
                 hr_preds[:, 0, :, :] = cu.softplus(hr_preds[:, 0, :, :])
                 hr_preds[:, 2, :, :] = cu.softplus(hr_preds[:, 2, :, :], c=0) + hr_preds[:, 1, :, :]
            for hr_pred in hr_preds:
                hr_pred = hr_pred.squeeze()
                hr_pred_pr = hr_pred[0]
                hr_pred_tasmin = hr_pred[1]
                hr_pred_tasmax = hr_pred[2]
                _, psdvals_pr = psd(hr_pred_pr)
                _, psdvals_tasmin = psd(hr_pred_tasmin)
                _, psdvals_tasmax = psd(hr_pred_tasmax)
                psd_pr.append(psdvals_pr)
                psd_tasmin.append(psdvals_tasmin)
                psd_tasmax.append(psdvals_tasmax)
                
        elif dataloader.dataset.type == "lrinterp_to_residuals" or dataloader.dataset.type == "lr_to_residuals":
            inputs, lrinterp = (batch['inputs'].to(device), batch['lrinterp'])
            preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
            hr_preds = dataloader.dataset.residual_to_hr(preds.detach().cpu(), lrinterp)
            if transfo:
                 hr_preds[:, 0, :, :] = cu.softplus(hr_preds[:, 0, :, :])
                 hr_preds[:, 2, :, :] = cu.softplus(hr_preds[:, 2, :, :], c=0) + hr_preds[:, 1, :, :]
            for hr_pred in hr_preds:
                hr_pred = hr_pred.squeeze()
                hr_pred_pr = hr_pred[0]
                hr_pred_tasmin = hr_pred[1]
                hr_pred_tasmax = hr_pred[2]
                _, psdvals_pr = psd(hr_pred_pr)
                _, psdvals_tasmin = psd(hr_pred_tasmin)
                _, psdvals_tasmax = psd(hr_pred_tasmax)
                psd_pr.append(psdvals_pr)
                psd_tasmin.append(psdvals_tasmin)
                psd_tasmax.append(psdvals_tasmax)

        psd_pr = np.array(psd_pr).mean(axis=0)
        psd_tasmin = np.array(psd_tasmin).mean(axis=0)
        psd_tasmax = np.array(psd_tasmax).mean(axis=0)

        return psd_pr, psd_tasmin, psd_tasmax
    
def compute_psd_over_groundtruth(hr):

    """
    Function to compute the power spectral density of the groundtruth (torch tensor).
    """

    psd_pr = []
    psd_tasmin = []
    psd_tasmax = []

    for sample in hr:
                sample = sample.squeeze()
                sample_pr = sample[0]
                sample_tasmin = sample[1]
                sample_tasmax = sample[2]
                _, psdvals_pr = psd(sample_pr)
                _, psdvals_tasmin = psd(sample_tasmin)
                _, psdvals_tasmax = psd(sample_tasmax)
                psd_pr.append(psdvals_pr)
                psd_tasmin.append(psdvals_tasmin)
                psd_tasmax.append(psdvals_tasmax)

    psd_pr = np.array(psd_pr).mean(axis=0)
    psd_tasmin = np.array(psd_tasmin).mean(axis=0)
    psd_tasmax = np.array(psd_tasmax).mean(axis=0)

    return psd_pr, psd_tasmin, psd_tasmax
