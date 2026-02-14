import numpy as np
import pytest

from SpatialOmics_Foundation import peak_filter_refactored as pf


def test_to_array_none_and_empty_and_bad_shape():
    """Test _to_array with None, empty list, and incorrect shapes"""
    assert pf._to_array(None).shape == (0, 2)
    assert pf._to_array([]).shape == (0, 2)

    with pytest.raises(ValueError):
        pf._to_array([1, 2, 3])  # ndim is wrong

    with pytest.raises(ValueError):
        pf._to_array([[1, 2, 3]])  # (N,3) is wrong


def test_set_mz_range_noop_and_swap_and_infer():
    """Test m/z range filtering with various parameter combinations"""
    peaks = np.array([[100.0, 1.0], [200.0, 2.0], [300.0, 3.0]])

    # both None -> infer from data, equal to no-op (keep all)
    out = pf.set_mz_range(peaks, None, None)
    assert out.shape[0] == 3

    # max < min -> swap branch
    out2 = pf.set_mz_range(peaks, 250.0, 150.0)
    # keep in [150,250] => only 200
    assert out2.shape[0] == 1
    assert out2[0, 0] == 200.0

    # inclusive range: boundaries should be included
    out3 = pf.set_mz_range(peaks, 100.0, 200.0)
    assert out3.shape[0] == 2
    assert set(out3[:, 0].tolist()) == {100.0, 200.0}


def test_remove_precursor_peak_invalid_unit_raises():
    """Test that invalid unit parameter raises ValueError"""
    peaks = np.array([[100.0, 1.0]])
    with pytest.raises(ValueError):
        pf.remove_precursor_peak(peaks, 1.0, unit="meter")  # unit not in {"Da","ppm"}


def test_remove_precursor_peak_branch_no_precursor_and_with_precursor_Da():
    """Test precursor peak removal with and without precursor_mz"""
    peaks = np.array([[99.0, 1.0], [100.0, 2.0], [101.0, 3.0]])

    # branch: precursor_mz missing -> do nothing
    out_no = pf.remove_precursor_peak(peaks, tol=0.5, unit="Da", precursor_mz=None)
    assert out_no.shape[0] == 3

    # with precursor: |mz-100|<=0.5 will be removed, so mz=100 is removed
    out_yes = pf.remove_precursor_peak(peaks, tol=0.5, unit="Da", precursor_mz=100.0)
    assert set(out_yes[:, 0].tolist()) == {99.0, 101.0}


def test_remove_precursor_peak_ppm_unit():
    """Test precursor peak removal with ppm unit"""
    peaks = np.array([[998.0, 1.0], [1000.0, 2.0], [1002.0, 3.0]])
    
    # ppm: tol=1000 ppm = 0.1% -> threshold = 1000 * 1000 * 1e-6 = 1.0 Da
    # |mz-1000| <= 1.0 -> only 1000.0 is removed (998 and 1002 are > 1.0 away)
    out = pf.remove_precursor_peak(peaks, tol=1000, unit="ppm", precursor_mz=1000.0)
    assert set(out[:, 0].tolist()) == {998.0, 1002.0}


def test_scale_intensity_none_root_log_rank_and_errors():
    """Test various intensity scaling methods"""
    peaks = np.array([[100.0, -1.0], [200.0, 4.0], [300.0, 9.0]])

    # scaling None: Negative values will be clipped to 0 (but not discarded)
    out_none = pf.scale_intensity(peaks, scaling=None)
    assert out_none[0, 1] == 0.0

    # root: sqrt(4)=2, sqrt(9)=3
    out_root = pf.scale_intensity(peaks, scaling="root", degree=2)
    assert abs(out_root[1, 1] - 2.0) < 1e-9
    assert abs(out_root[2, 1] - 3.0) < 1e-9

    # log: log2(inten+1)
    out_log = pf.scale_intensity(np.array([[100.0, 3.0]]), scaling="log", base=2)
    assert abs(out_log[0, 1] - 2.0) < 1e-9  # log2(4)=2

    # rank: The one with the highest intensity has the highest rank
    out_rank = pf.scale_intensity(
        np.array([[1.0, 10.0], [2.0, 1.0], [3.0, 5.0]]), 
        scaling="rank"
    )
    # The peak with intensity=10 should have the highest rank
    max_rank_idx = np.argmax(out_rank[:, 1])
    assert out_rank[max_rank_idx, 0] == 1.0

    # max_rank < n should raise an error
    with pytest.raises(ValueError):
        pf.scale_intensity(
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), 
            scaling="rank", 
            max_rank=2
        )

    # unsupported scaling should raise an error
    with pytest.raises(ValueError):
        pf.scale_intensity(peaks, scaling="weird")


def test_scale_intensity_with_max_intensity():
    """Test scaling with max_intensity normalization"""
    peaks = np.array([[100.0, 50.0], [200.0, 100.0]])
    
    # Normalize max to 10.0
    out = pf.scale_intensity(peaks, scaling=None, max_intensity=10.0)
    assert abs(out[1, 1] - 10.0) < 1e-9  # max should be 10
    assert abs(out[0, 1] - 5.0) < 1e-9   # half of max should be 5


def test_filter_intensity_threshold_max_peaks_and_sorting():
    """Test intensity filtering with threshold and max_peaks"""
    # Intensities with <=0 will be discarded first
    peaks = np.array([
        [300.0, 10.0],
        [100.0, 0.0],
        [200.0, 1.0],
        [150.0, 5.0],
    ])

    # max intensity=10, threshold=0.2*10=2, and we keep those >2: 10 and 5
    out = pf.filter_intensity(peaks, min_intensity=0.2, max_peaks=None)
    assert set(out[:, 0].tolist()) == {150.0, 300.0}

    # max_peaks truncation branch
    out2 = pf.filter_intensity(peaks, min_intensity=0.0, max_peaks=1)
    assert out2.shape[0] == 1
    assert out2[0, 1] == 10.0  # The strongest peak

    # Result stability: sorted by mz
    out3 = pf.filter_intensity(peaks, min_intensity=0.0, max_peaks=None)
    assert np.all(out3[:, 0] == np.sort(out3[:, 0]))


def test_discard_low_quality_branch():
    """Test discarding spectra with too few peaks"""
    peaks = np.array([[100.0, 1.0], [200.0, 2.0]])
    
    # Too few peaks -> discard (return None)
    out = pf.discard_low_quality(peaks, min_peaks=3)
    assert out is None

    # Enough peaks -> keep
    out2 = pf.discard_low_quality(peaks, min_peaks=2)
    assert out2 is not None
    assert out2.shape[0] == 2


def test_scale_to_unit_norm_zero_norm_and_nonzero():
    """Test L2 normalization with zero and non-zero intensities"""
    # All zero intensities -> norm=0 branch: remain unchanged
    peaks0 = np.array([[100.0, 0.0], [200.0, 0.0]])
    out0 = pf.scale_to_unit_norm(peaks0)
    assert np.all(out0[:, 1] == 0.0)

    # Non-zero: L2 norm should be approximately 1
    peaks = np.array([[100.0, 3.0], [200.0, 4.0]])
    out = pf.scale_to_unit_norm(peaks)
    norm = np.sqrt(np.sum(out[:, 1] ** 2))
    assert abs(norm - 1.0) < 1e-9


def test_apply_preprocessing_pipeline_basic():
    """Test basic preprocessing pipeline execution"""
    peaks = np.array([[100.0, 1.0], [200.0, 2.0], [300.0, 3.0]])
    
    steps = [
        {'function': 'set_mz_range', 'params': {'min_mz': 150.0, 'max_mz': 250.0}},
        {'function': 'filter_intensity', 'params': {'min_intensity': 0.0}},
    ]
    
    out = pf.apply_preprocessing_pipeline(peaks, steps)
    assert out is not None
    assert out.shape[0] == 1  # Only peak at 200.0 is in range [150, 250]
    assert out[0, 0] == 200.0


def test_apply_preprocessing_pipeline_with_precursor():
    """Test pipeline with functions that require precursor_mz"""
    peaks = np.array([[100.0, 1.0], [101.0, 2.0], [150.0, 3.0]])
    
    steps = [
        {'function': 'remove_precursor_peak', 'params': {'tol': 0.1, 'unit': 'Da'}},
        {'function': 'set_mz_range', 'params': {'min_mz': 100.0, 'max_mz': 200.0}},
    ]
    
    # With precursor_mz=None, remove_precursor_peak is a no-op
    out = pf.apply_preprocessing_pipeline(peaks, steps, precursor_mz=None)
    assert out is not None
    assert out.shape[0] == 3
    
    # With precursor_mz=100.0, peak at 100.0 should be removed
    out2 = pf.apply_preprocessing_pipeline(peaks, steps, precursor_mz=100.0)
    assert out2 is not None
    assert 100.0 not in out2[:, 0]


def test_apply_preprocessing_pipeline_returns_none():
    """Test that pipeline returns None when discard_low_quality triggers"""
    peaks = np.array([[100.0, 1.0], [200.0, 2.0]])
    
    steps = [
        {'function': 'discard_low_quality', 'params': {'min_peaks': 5}},
    ]
    
    out = pf.apply_preprocessing_pipeline(peaks, steps)
    assert out is None


def test_apply_preprocessing_pipeline_invalid_function():
    """Test that invalid function name raises error"""
    peaks = np.array([[100.0, 1.0]])
    
    steps = [
        {'function': 'nonexistent_function', 'params': {}},
    ]
    
    with pytest.raises(ValueError):
        pf.apply_preprocessing_pipeline(peaks, steps)


def test_full_preprocessing_pipeline():
    """Test complete preprocessing pipeline with all functions"""
    peaks = np.array([
        [50.0, 10.0],
        [100.0, 20.0],
        [150.0, 100.0],  # precursor
        [200.0, 50.0],
        [300.0, 5.0],
    ])
    
    steps = [
        {'function': 'set_mz_range', 'params': {'min_mz': 100.0, 'max_mz': 250.0}},
        {'function': 'remove_precursor_peak', 'params': {'tol': 1.0, 'unit': 'Da'}},
        {'function': 'scale_intensity', 'params': {'scaling': 'root', 'max_intensity': 1.0}},
        {'function': 'filter_intensity', 'params': {'min_intensity': 0.1, 'max_peaks': 10}},
        {'function': 'discard_low_quality', 'params': {'min_peaks': 1}},
        {'function': 'scale_to_unit_norm', 'params': {}},
    ]
    
    out = pf.apply_preprocessing_pipeline(peaks, steps, precursor_mz=150.0)
    
    assert out is not None
    assert out.shape[0] > 0
    # Verify it's normalized
    norm = np.sqrt(np.sum(out[:, 1] ** 2))
    assert abs(norm - 1.0) < 1e-9
    # Verify 150.0 was removed
    assert 150.0 not in out[:, 0]
    # Verify 50.0 was removed (out of range)
    assert 50.0 not in out[:, 0]
