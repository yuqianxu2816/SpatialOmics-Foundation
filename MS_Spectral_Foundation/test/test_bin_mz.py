import numpy as np
import pytest
import torch


from SpatialOmics_Foundation import bin_mz as bm

def _make_npz(tmp_path, peaks_mz_list, name="spectra_filtered.npz"):
    # peaks_mz must be an object array, each item is (Ni,) float
    peaks_mz = np.empty((len(peaks_mz_list),), dtype=object)
    for i, arr in enumerate(peaks_mz_list):
        peaks_mz[i] = np.asarray(arr, dtype=float)
    p = tmp_path / name
    np.savez_compressed(p, peaks_mz=peaks_mz)
    return p


def test_bin_mz_all_invalid_returns_zeros(tmp_path):
    npz = _make_npz(tmp_path, [[0.0, 0.0], [0.0]])
    out = bm.bin_mz(str(npz), bin_size=10.0, mz_min=50.0, mz_max=150.0, is_normalized=False)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long
    # All m/z <= 0, so valid_mask.any() should be False, and out variable should be all 0
    assert torch.all(out == 0)


def test_not_normalized_bin_mz__binning_and_clamp_and_padding(tmp_path):
    # Two spectra: the second is shorter, padding is automatically filled with 0, and finally bins[~valid_mask]=0 
    # Set parameters: mz_min=50, mz_max=150, bin_size=10 => n_bins=10
    # mz=50 -> floor((50-50)/10)=0
    # mz=55 -> floor((55-50)/10)=0
    # mz=149 -> floor((149-50)/10)=9
    # mz=200 -> clamp to 150 -> floor((150-50)/10)=10 -> then clamp to 9
    npz = _make_npz(tmp_path, [[50.0, 55.0, 149.0, 200.0], [60.0]])
    out = bm.bin_mz(str(npz), bin_size=10.0, mz_min=50.0, mz_max=150.0, is_normalized=False)

    assert out.shape[0] == 2
    assert out.shape[1] == 4  # pad to max_len=4

    row0 = out[0].tolist()
    assert row0[0] == 0
    assert row0[1] == 0
    assert row0[2] == 9
    assert row0[3] == 9  # clamped

    row1 = out[1].tolist()
    assert row1[0] == 1  
    assert row1[1] == 0  # pad to 0 since valid mask is false
    assert row1[2] == 0
    assert row1[3] == 0


def test_normalized_bin_mz(tmp_path):
    # is_normalized=True: bins = floor(clamped * n_bins) 
    # Ex: clamped = 0.05
    # Ex: bins = torch.floor(0.05 × 10) = floor(0.5) = 0
    # n_bins = ceil((mz_max-mz_min)/bin_size)
    # mz_min=0, mz_max=1, bin_size=0.1 => n_bins=10
    # 0.0 -> 0, 0.05 -> 0, 0.99 -> 9, 1.0 -> floor(10)=10 -> clamp-> 9
    npz = _make_npz(tmp_path, [[0.0, 0.05, 0.99, 1.0]])
    out = bm.bin_mz(str(npz), bin_size=0.1, mz_min=0.0, mz_max=1.0, is_normalized=True)

    assert out.shape == (1, 4)
    assert out[0, 0].item() == 0
    assert out[0, 1].item() == 0
    assert out[0, 2].item() == 9
    assert out[0, 3].item() == 9
