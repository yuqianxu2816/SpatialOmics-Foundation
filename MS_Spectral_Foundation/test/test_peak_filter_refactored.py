import numpy as np
import pytest



from MS_Spectral_Foundation import peak_filter_refactored as pf


def test_to_array_none_and_empty_and_bad_shape():
    assert pf._to_array(None).shape == (0, 2)
    assert pf._to_array([]).shape == (0, 2)

    with pytest.raises(ValueError):
        pf._to_array([1, 2, 3])  # testing whether is a 2D array

    with pytest.raises(ValueError):
        pf._to_array([[1, 2, 3]])  # is a two D array, but (N,3) is wrong, should be (N, 2), two columns for mz and intensity, not three


def test_set_mz_range_noop_and_swap_and_infer():
    peaks = np.array([[100.0, 1.0], [200.0, 2.0], [300.0, 3.0]])

    # mz_min and mz_max both None -> infer from data branch -> keep all 3 peaks
    out = pf.set_mz_range(peaks, None, None)
    assert out.shape[0] == 3

    # max < min -> swap
    out2 = pf.set_mz_range(peaks, 250.0, 150.0)
    # keep in [150,250] => only 200
    assert out2.shape[0] == 1
    assert out2[0, 0] == 200.0

    # inclusive range:
    out3 = pf.set_mz_range(peaks, 100.0, 200.0)
    assert out3.shape[0] == 2
    assert set(out3[:, 0].tolist()) == {100.0, 200.0}


def test_remove_precursor_peak_invalid_unit_raises():
    peaks = np.array([[100.0, 1.0]])
    with pytest.raises(ValueError):
        pf.remove_precursor_peak(peaks, None, 1.0, unit="meter")  # unit: not in {"Da","ppm"}


def test_remove_precursor_peak_branch_no_precursor_and_with_precursor_Da():
    peaks = np.array([[99.0, 1.0], [100.0, 2.0], [101.0, 3.0]])

    # precursor_mz missing -> keep all 3 peaks (unable to recognize precursor peak)
    out_no = pf.remove_precursor_peak(peaks, precursor_mz=None, tol=0.5, unit="Da")
    assert out_no.shape[0] == 3

    # with precursor: |mz-100|<=0.5 will be removed, which mz=100 is removed
    out_yes = pf.remove_precursor_peak(peaks, precursor_mz=100.0, tol=0.5, unit="Da")
    assert set(out_yes[:, 0].tolist()) == {99.0, 101.0}


def test_scale_intensity_none_root_log_rank_and_errors():
    peaks = np.array([[100.0, -1.0], [200.0, 4.0], [300.0, 9.0]])

    # no scaling: -1 -> 0 
    out_none = pf.scale_intensity(peaks, None)
    assert out_none[0, 1] == 0.0

    # root: sqrt(4)=2
    out_root = pf.scale_intensity(peaks, "root", degree=2)
    assert abs(out_root[1, 1] - 2.0) < 1e-9
    assert abs(out_root[2, 1] - 3.0) < 1e-9

    # log: log2(inten+1)
    out_log = pf.scale_intensity(np.array([[100.0, 3.0]]), "log", base=2)
    assert abs(out_log[0, 1] - 2.0) < 1e-9

    # rank: highest intensity has the highest rank
    out_rank = pf.scale_intensity(np.array([[1.0, 10.0], [2.0, 1.0], [3.0, 5.0]]), "rank")
    # The peak with intensity=10 should have the highest rank
    max_rank_idx = np.argmax(out_rank[:, 1])
    assert out_rank[max_rank_idx, 0] == 1.0

    with pytest.raises(ValueError):
        pf.scale_intensity(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), "rank", max_rank=2)

    with pytest.raises(ValueError):
        pf.scale_intensity(peaks, "weird")


def test_filter_intensity_threshold_max_peaks_and_sorting():
    # Intensities with <=0 will be discarded first 
    peaks = np.array([
        [300.0, 10.0],
        [100.0, 0.0],
        [200.0, 1.0],
        [150.0, 5.0],
    ])

    # keep intensities > min_intensity * max_intensity,
    # 0.2*10=2.0, so only 10 and 5 are kept
    out = pf.filter_intensity(peaks, min_intensity=0.2, max_peaks=None)
    assert set(out[:, 0].tolist()) == {150.0, 300.0}

    # max_peaks=1: only keep the strongest peak of intensity = 10
    out2 = pf.filter_intensity(peaks, min_intensity=0.0, max_peaks=1)
    assert out2.shape[0] == 1
    assert out2[0, 1] == 10.0 

    # sorted by mz
    out3 = pf.filter_intensity(peaks, min_intensity=0.0, max_peaks=None)
    assert np.all(out3[:, 0] == np.sort(out3[:, 0]))


def test_discard_low_quality_branch():
    # At least 3 peaks are required, so 2 peaks should be discarded
    peaks = np.array([[100.0, 1.0], [200.0, 2.0]])
    assert pf.discard_low_quality(peaks, min_peaks=3) is None

    # At least 2 peaks are required, so 2 peaks should be kept
    out = pf.discard_low_quality(peaks, min_peaks=2)
    assert out is not None
    assert out.shape[0] == 2


def test_scale_to_unit_norm_zero_norm_and_nonzero():
    # All zero intensities -> norm=0
    # L2 norm = srqt(0^2+0^2)=0 -> return original peaks without scaling
    peaks0 = np.array([[100.0, 0.0], [200.0, 0.0]])
    out0 = pf._scale_to_unit_norm(peaks0)
    assert np.all(out0[:, 1] == 0.0)

    # Non-zero: L2 norm should be 1
    # norm = sqrt(3^2+4^2)=5 -> scaled intensities should be 3/5=0.6 and 4/5=0.8
    # sqrt(0.6^2+0.8^2)=1 -> unit norm achieved
    peaks = np.array([[100.0, 3.0], [200.0, 4.0]])
    out = pf._scale_to_unit_norm(peaks)
    norm = np.sqrt(np.sum(out[:, 1] ** 2))
    assert abs(norm - 1.0) < 1e-9


def test_apply_preprocessing_pipeline_typeerror_fallback_branch():
    # pipeline contain both: functions require precursor_mz (2 args) and not require (1 arg)
    peaks = np.array([[100.0, 1.0], [101.0, 2.0]])

    preprocessing_config = {
        "remove_precursor_peak": {"tol": 0.1, "unit": "Da"},  # (peaks, precursor_mz)
        "set_mz_range": {"min_mz": 100.0, "max_mz": 200.0},   # (peaks)
    }

    out = pf.apply_preprocessing_pipeline(peaks, preprocessing_config, precursor_mz=None)
    # When precursor=None, remove_precursor_peak does nothing; set_mz_range keeps both peaks
    assert out is not None
    assert out.shape[0] == 2
