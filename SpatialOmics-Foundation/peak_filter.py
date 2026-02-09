# spatialomics_foundation/peak_filter.py
from __future__ import annotations

from typing import Callable, Optional, Literal
import numpy as np

Unit = Literal["Da", "ppm"]
ScaleMethod = Literal["root", "log", "none"]

def _to_array(peaks) -> np.ndarray:
    """Convert peaks to float array of shape (N,2)."""
    if peaks is None:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(peaks, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("peaks must have shape (N,2): [[mz,intensity], ...]")
    return arr

def set_mz_range(min_mz: float, max_mz: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Keep peaks with mz in [min_mz, max_mz).
    """
    def _fn(peaks: np.ndarray) -> np.ndarray:
        p = _to_array(peaks)
        if p.shape[0] == 0:
            return p
        mz = p[:, 0]
        keep = (mz >= min_mz) & (mz < max_mz)
        return p[keep]
    return _fn

def remove_precursor_peak(tol: float, unit: Unit = "Da") -> Callable[[np.ndarray, Optional[float]], np.ndarray]:
    """
    Remove peaks around precursor_mz within tolerance.
    - unit="Da": |mz - precursor| <= tol
    - unit="ppm": |mz - precursor| <= precursor * tol * 1e-6
    Precursor m/z 通常从 meta['PEPMASS'] 来。
    """
    def _fn(peaks: np.ndarray, precursor_mz: Optional[float] = None) -> np.ndarray:
        p = _to_array(peaks)
        if p.shape[0] == 0:
            return p
        if precursor_mz is None or (isinstance(precursor_mz, float) and np.isnan(precursor_mz)):
            # 分支：没有 precursor 信息 -> 不做删除
            return p

        mz = p[:, 0]
        if unit == "Da":
            thr = tol
        elif unit == "ppm":
            thr = float(precursor_mz) * tol * 1e-6
        else:
            raise ValueError("unit must be 'Da' or 'ppm'")

        keep = np.abs(mz - float(precursor_mz)) > thr
        return p[keep]
    return _fn

def scale_intensity(method: ScaleMethod = "root", param: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Intensity scaling.
    - root: intensity := intensity ** (1/2) when param=1 (sqrt)
            more generally intensity := intensity ** (1/(param+1))  (param>=1)
    - log:  intensity := log(1 + param*intensity)
    - none: unchanged
    """
    def _fn(peaks: np.ndarray) -> np.ndarray:
        p = _to_array(peaks).copy()
        if p.shape[0] == 0:
            return p

        inten = p[:, 1]
        # 负值强度先裁到 0（你也可以选择直接丢弃；这里更鲁棒）
        inten = np.maximum(inten, 0.0)

        if method == "none":
            p[:, 1] = inten
            return p
        if method == "root":
            # param=1 -> sqrt
            expo = 1.0 / (float(param) + 1.0) if param >= 1 else 0.5
            p[:, 1] = np.power(inten, expo)
            return p
        if method == "log":
            p[:, 1] = np.log1p(float(param) * inten)
            return p

        raise ValueError("method must be 'root', 'log', or 'none'")
    return _fn

def filter_intensity(min_intensity: float, max_peaks: Optional[int]) -> Callable[[np.ndarray], np.ndarray]:
    """
    - remove peaks with intensity < min_intensity
    - keep at most max_peaks peaks by intensity (top-k)
    """
    def _fn(peaks: np.ndarray) -> np.ndarray:
        p = _to_array(peaks)
        if p.shape[0] == 0:
            return p

        keep = p[:, 1] >= float(min_intensity)
        p = p[keep]
        if p.shape[0] == 0:
            return p

        if max_peaks is not None and p.shape[0] > int(max_peaks):
            idx = np.argsort(p[:, 1])[::-1][: int(max_peaks)]
            p = p[idx]

        #（可选）按 mz 排序，利于稳定性
        p = p[np.argsort(p[:, 0])]
        return p
    return _fn

def discard_low_quality(min_peaks: int) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    """
    If remaining peaks < min_peaks, discard the spectrum (return None).
    """
    def _fn(peaks: np.ndarray) -> Optional[np.ndarray]:
        p = _to_array(peaks)
        if p.shape[0] < int(min_peaks):
            return None
        return p
    return _fn

def _scale_to_unit_norm(peaks: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize intensity vector to unit norm.
    If norm=0 -> return unchanged (all zero intensities).
    """
    p = _to_array(peaks).copy()
    if p.shape[0] == 0:
        return p
    inten = np.maximum(p[:, 1], 0.0)
    norm = np.sqrt(np.sum(inten * inten))
    if norm <= eps:
        p[:, 1] = inten
        return p
    p[:, 1] = inten / norm
    return p

def apply_preprocessing_pipeline(
    peaks: np.ndarray,
    pipeline: list[Callable],
    precursor_mz: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Apply the list of preprocessing functions in order.
    Some steps may need precursor_mz; we support that here.
    """
    x: Optional[np.ndarray] = peaks
    for fn in pipeline:
        if x is None:
            return None

        # 支持 remove_precursor_peak 这种需要 precursor_mz 的函数
        try:
            x = fn(x, precursor_mz)  # type: ignore[misc]
        except TypeError:
            x = fn(x)  # type: ignore[misc]
    return x





# ====== Script-only runner (no argparse) ======
import os
import numpy as np
from spatialomics_foundation.mgf_parse import load_spectra_npz, save_spectra_npz

# ✅ 改成你实际输出位置
IN_NPZ  = "output/spectra_raw.npz"
OUT_NPZ = "output/spectra_filtered.npz"

# ✅ 你的 filter 参数
MIN_MZ = 100.0
MAX_MZ = 2000.0
MIN_INTENSITY = 1e-12     # 等价于 “>0”
MAX_PEAKS = 200
MIN_PEAKS = 5
REMOVE_PRECURSOR_TOL = 1.5
PRECURSOR_UNIT = "Da"     # "Da" or "ppm"

def main():
    spectra = load_spectra_npz(IN_NPZ)
    print(f"[peak_filter] loaded spectra: {len(spectra)} from {IN_NPZ}")

    pipeline = [
        set_mz_range(MIN_MZ, MAX_MZ),
        remove_precursor_peak(REMOVE_PRECURSOR_TOL, PRECURSOR_UNIT),
        scale_intensity("root", 1),
        filter_intensity(MIN_INTENSITY, MAX_PEAKS),
        discard_low_quality(MIN_PEAKS),
        _scale_to_unit_norm,
    ]

    filtered = []
    dropped = 0

    for sp in spectra:
        precursor_mz = sp["meta"].get("PEPMASS", None)
        peaks = np.array(sp["peaks"], dtype=float)

        out = apply_preprocessing_pipeline(peaks, pipeline, precursor_mz=precursor_mz)
        if out is None:
            dropped += 1
            continue

        # 保存回同一种 Spectrum 格式（后续 binning / SSL 最好接）
        sp2 = {
            "meta": sp["meta"],
            "peaks": list(map(tuple, out.tolist())),  # list[(mz,inten)]
        }
        filtered.append(sp2)

    print(f"[peak_filter] kept={len(filtered)} dropped={dropped}")

    save_spectra_npz(filtered, OUT_NPZ)
    print(f"[peak_filter] saved -> {OUT_NPZ}")

    # sanity check: reload and print one
    s2 = load_spectra_npz(OUT_NPZ)
    print(f"[peak_filter] reload check: {len(s2)} spectra")
    if len(s2) > 0:
        print("[peak_filter] spectrum[0].meta =", s2[0]["meta"])
        print("[peak_filter] spectrum[0].num_peaks =", len(s2[0]["peaks"]))

if __name__ == "__main__":
    main()
