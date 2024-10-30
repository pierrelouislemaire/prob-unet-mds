import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image

def mae(y_true, y_pred, sample_weigths=None):
    mae = np.average(np.abs(y_true - y_pred), axis=0, weights=sample_weigths)
    return mae

def rmse(y_true, y_pred, sample_weights=None):
    rmse = np.sqrt(np.average((y_pred-y_true)**2, axis=0, weights=sample_weights))
    return rmse

def crps(y_true, y_pred, sample_weights=None):
    # CRPS PWM
    if torch.is_tensor(y_true):
        y_true = np.array(y_true)
    if torch.is_tensor(y_pred):
        y_pred = np.array(y_pred)
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)
    if len(y_pred.shape) == 3:
        return absolute_error
    y_pred = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * np.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples
    per_obs_crps = absolute_error + b0 - 2 * b1
    return np.average(per_obs_crps, weights=sample_weights)

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
def compute_psd_over_loader(model, dataloader, device):
    
    model.eval()

    psd_pr = []
    psd_tasmin = []
    psd_tasmax = []

    for i, batch in enumerate(dataloader):

        if dataloader.dataset.type == "lr_to_hr" or dataloader.dataset.type == "lrinterp_to_hr":
            inputs, stand_stats = (batch['inputs'].to(device), batch['stand_stats'])
            preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
            hr_preds = dataloader.dataset.invstand_residual(preds.detach().cpu(), stand_stats)
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
            inputs, lrinterp, stand_stats = (batch['inputs'].to(device), batch['lrinterp'], batch['stand_stats'])
            preds = model(inputs, batch["timestamps"].unsqueeze(dim=1).to(device))
            hr_preds = dataloader.dataset.residual_to_hr(preds.detach().cpu(), lrinterp, stand_stats)
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

if __name__ == "__main__":

    image = Image.open("clouds.png")
    image = np.array(image)/255.

    kvals, psdvals = psd(torch.tensor(image))
    
    plt.loglog(kvals, psdvals)
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.tight_layout()
    plt.savefig("cloud_power_spectrum.png", dpi = 300, bbox_inches = "tight")