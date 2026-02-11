import numpy as np
import pytest



from SpatialOmics_Foundation import peak_filter as pf


def test_to_array_none_and_empty_and_bad_shape():
    assert pf._to_array(None).shape == (0, 2)
    assert pf._to_array([]).shape == (0, 2)

    with pytest.raises(ValueError):
        pf._to_array([1, 2, 3])  # ndim is worng

    with pytest.raises(ValueError):
        pf._to_array([[1, 2, 3]])  # (N,3) is wrong


def test_set_mz_range_noop_and_swap_and_infer():
    peaks = np.array([[100.0, 1.0], [200.0, 2.0], [300.0, 3.0]])

    # both None -> infer from data，equal to no-op（keep all）
    fn = pf.set_mz_range(None, None)
    out = fn(peaks)
    assert out.shape[0] == 3

    # max < min -> swap branch
    fn2 = pf.set_mz_range(250.0, 150.0)
    out2 = fn2(peaks)
    # keep in [150,250] => only 200
    assert out2.shape[0] == 1
    assert out2[0, 0] == 200.0

    # inclusive range：:contentReference[oaicite:8]{index=8}
    fn3 = pf.set_mz_range(100.0, 200.0)
    out3 = fn3(peaks)
    assert out3.shape[0] == 2
    assert set(out3[:, 0].tolist()) == {100.0, 200.0}


def test_remove_precursor_peak_invalid_unit_raises():
    with pytest.raises(ValueError):
        pf.remove_precursor_peak(1.0, unit="meter")  # unit branch :contentReference[oaicite:9]{index=9}


def test_remove_precursor_peak_branch_no_precursor_and_with_precursor_Da():
    peaks = np.array([[99.0, 1.0], [100.0, 2.0], [101.0, 3.0]])

    fn = pf.remove_precursor_peak(tol=0.5, unit="Da")

    # branch: precursor_mz missing -> no-op :contentReference[oaicite:10]{index=10}
    out_no = fn(peaks, precursor_mz=None)
    assert out_no.shape[0] == 3

    # with precursor: |mz-100|<=0.5 will be removed => mz=100
    out_yes = fn(peaks, precursor_mz=100.0)
    assert set(out_yes[:, 0].tolist()) == {99.0, 101.0}


def test_scale_intensity_none_root_log_rank_and_errors():
    peaks = np.array([[100.0, -1.0], [200.0, 4.0], [300.0, 9.0]])

    # scaling None: Negative values will be clipped to 0 (but not discarded) :contentReference[oaicite:11]{index=11}
    fn_none = pf.scale_intensity(None)
    out_none = fn_none(peaks)
    assert out_none[0, 1] == 0.0

    # root: sqrt(4)=2, sqrt(9)=3 :contentReference[oaicite:12]{index=12}
    fn_root = pf.scale_intensity("root", degree=2)
    out_root = fn_root(peaks)
    assert abs(out_root[1, 1] - 2.0) < 1e-9
    assert abs(out_root[2, 1] - 3.0) < 1e-9

    # log: log2(inten+1) :contentReference[oaicite:13]{index=13}
    fn_log = pf.scale_intensity("log", base=2)
    out_log = fn_log(np.array([[100.0, 3.0]]))
    assert abs(out_log[0, 1] - 2.0) < 1e-9  # log2(4)=2

    # rank: The one with the highest intensity has the highest rank :contentReference[oaicite:14]{index=14}
    fn_rank = pf.scale_intensity("rank")
    out_rank = fn_rank(np.array([[1.0, 10.0], [2.0, 1.0], [3.0, 5.0]]))
    # The peak with intensity=10 should have the highest rank
    max_rank_idx = np.argmax(out_rank[:, 1])
    assert out_rank[max_rank_idx, 0] == 1.0

    # max_rank < n -> should raise an error :contentReference[oaicite:15]{index=15}
    with pytest.raises(ValueError):
        pf.scale_intensity("rank", max_rank=2)(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))

    # unsupported scaling -> should raise an error :contentReference[oaicite:16]{index=16}
    with pytest.raises(ValueError):
        pf.scale_intensity("weird")(peaks)


def test_filter_intensity_threshold_max_peaks_and_sorting():
    # Intensities with <=0 will be discarded first :contentReference[oaicite:17]{index=17}
    peaks = np.array([
        [300.0, 10.0],
        [100.0, 0.0],
        [200.0, 1.0],
        [150.0, 5.0],
    ])

    fn = pf.filter_intensity(min_intensity=0.2, max_peaks=None)
    # max intensity=10, threshold=2 => keep those >2: 10 and 5
    out = fn(peaks)
    assert set(out[:, 0].tolist()) == {150.0, 300.0}

    # max_peaks truncation branch :contentReference[oaicite:18]{index=18}
    fn2 = pf.filter_intensity(min_intensity=0.0, max_peaks=1)
    out2 = fn2(peaks)
    assert out2.shape[0] == 1
    assert out2[0, 1] == 10.0  # The strongest peak

    # Result stability: sorted by mz :contentReference[oaicite:19]{index=19}
    fn3 = pf.filter_intensity(min_intensity=0.0, max_peaks=None)
    out3 = fn3(peaks)
    assert np.all(out3[:, 0] == np.sort(out3[:, 0]))


def test_discard_low_quality_branch():
    peaks = np.array([[100.0, 1.0], [200.0, 2.0]])
    fn = pf.discard_low_quality(min_peaks=3)
    assert fn(peaks) is None  # ✅ Discard branch :contentReference[oaicite:20]{index=20}

    fn2 = pf.discard_low_quality(min_peaks=2)
    out = fn2(peaks)
    assert out is not None
    assert out.shape[0] == 2


def test_scale_to_unit_norm_zero_norm_and_nonzero():
    # All zero intensities -> norm=0 branch: remain unchanged :contentReference[oaicite:21]{index=21}
    peaks0 = np.array([[100.0, 0.0], [200.0, 0.0]])
    out0 = pf._scale_to_unit_norm(peaks0)
    assert np.all(out0[:, 1] == 0.0)

    # Non-zero: L2 norm should be approximately 1
    peaks = np.array([[100.0, 3.0], [200.0, 4.0]])
    out = pf._scale_to_unit_norm(peaks)
    norm = np.sqrt(np.sum(out[:, 1] ** 2))
    assert abs(norm - 1.0) < 1e-9


def test_apply_preprocessing_pipeline_typeerror_fallback_branch():
    # Let the pipeline contain both: functions that require precursor_mz (2 args) and those that do not (1 arg)
    peaks = np.array([[100.0, 1.0], [101.0, 2.0]])

    fn_need_prec = pf.remove_precursor_peak(tol=0.1, unit="Da")  # (peaks, precursor_mz)
    fn_no_prec = pf.set_mz_range(100.0, 200.0)                   # (peaks)

    out = pf.apply_preprocessing_pipeline(peaks, [fn_need_prec, fn_no_prec], precursor_mz=None)
    # When precursor=None, remove_precursor_peak is a no-op; set_mz_range keeps both peaks
    assert out is not None
    assert out.shape[0] == 2
