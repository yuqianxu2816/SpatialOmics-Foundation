# spatialomics_foundation/peak_filter_refactored.py
from __future__ import annotations

from typing import Optional, Literal
import numpy as np

Unit = Literal["Da", "ppm"]
ScaleMethod = Optional[Literal["root", "log", "rank", "none"]]


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


def set_mz_range(peaks: np.ndarray, min_mz: Optional[float] = None, max_mz: Optional[float] = None) -> np.ndarray:
    """
    Restrict m/z values to [min_mz, max_mz] (inclusive).
    - if both None: no-op
    - if one side None: infer from data
    - if max < min: swap
    """
    p = _to_array(peaks)
    if p.shape[0] == 0:
        return p

    mz = p[:, 0]
    lo = float(np.min(mz)) if min_mz is None else float(min_mz)
    hi = float(np.max(mz)) if max_mz is None else float(max_mz)
    if hi < lo:
        lo, hi = hi, lo

    keep = (mz >= lo) & (mz <= hi)   # inclusive
    return p[keep]


def remove_precursor_peak(peaks: np.ndarray, tol: float, unit: Unit = "Da", precursor_mz: Optional[float] = None) -> np.ndarray:
    """
    Remove peaks around precursor_mz within tolerance.
    - unit="Da": |mz - precursor| <= tol
    - unit="ppm": |mz - precursor| <= precursor * tol * 1e-6

    If precursor_mz is missing -> no-op (branch coverage).
    """
    if unit not in ("Da", "ppm"):
        raise ValueError("unit must be 'Da' or 'ppm'")

    p = _to_array(peaks)
    if p.shape[0] == 0:
        return p

    if precursor_mz is None or (isinstance(precursor_mz, float) and np.isnan(precursor_mz)):
        return p  # branch: no precursor info

    precursor_mz = float(precursor_mz)
    mz = p[:, 0]

    thr = tol if unit == "Da" else precursor_mz * float(tol) * 1e-6
    keep = np.abs(mz - precursor_mz) > float(thr)
    return p[keep]


def scale_intensity(
    peaks: np.ndarray,
    scaling: ScaleMethod = None,
    max_intensity: Optional[float] = None,
    degree: int = 2,
    base: int = 2,
    max_rank: Optional[int] = None,
) -> np.ndarray:
    """
    Scale peak intensities using various methods: root, log, or rank transformation.
    Optionally normalize intensities relative to the most intense peak to a specified maximum value.
    """
    p = _to_array(peaks).copy()
    if p.shape[0] == 0:
        return p

    inten = p[:, 1].astype(float)
    # The DDS base filter: remove zero/negative intensity (here clip to >=0, actual removal done by filter_intensity)
    inten = np.maximum(inten, 0.0)

    # scale relative to most intense peak (optional)
    if max_intensity is not None:
        mx = float(np.max(inten)) if inten.size > 0 else 0.0
        if mx > 0:
            inten = inten / mx * float(max_intensity)

    if scaling is None or scaling == "none":
        p[:, 1] = inten
        return p

    if scaling == "root":
        d = int(degree) if int(degree) > 0 else 2
        p[:, 1] = np.power(inten, 1.0 / d)
        return p

    if scaling == "log":
        b = float(base) if float(base) > 1 else 2.0
        p[:, 1] = np.log(inten + 1.0) / np.log(b)
        return p

    if scaling == "rank":
        # rank-transform: highest intensity gets max_rank
        n = inten.size
        if n == 0:
            p[:, 1] = inten
            return p
        mr = n if max_rank is None else int(max_rank)
        if mr < n:
            raise ValueError("max_rank should be >= number of peaks")
        order = np.argsort(inten)  # low -> high
        ranks = np.empty_like(inten, dtype=float)
        # smallest -> 1, largest -> mr (spread linearly)
        # simplest: map positions to [1..n] then scale to [1..mr]
        base_r = np.arange(1, n + 1, dtype=float)
        ranks[order] = base_r * (float(mr) / float(n))
        p[:, 1] = ranks
        return p

    raise ValueError("scaling must be one of: None, 'none', 'root', 'log', 'rank'")


def filter_intensity(peaks: np.ndarray, min_intensity: float = 0.0, max_peaks: Optional[int] = None) -> np.ndarray:
    """
    Filter out low-intensity peaks using a relative threshold (min_intensity * max_intensity).
    Optionally keep only the top max_peaks most intense peaks, then sort by m/z for stability.
    """
    p = _to_array(peaks)
    if p.shape[0] == 0:
        return p

    # DDS base filter: remove zero/negative
    p = p[p[:, 1] > 0.0]
    if p.shape[0] == 0:
        return p

    inten = p[:, 1]
    mx = float(np.max(inten))
    thr = float(min_intensity) * mx  # spectrum-style relative threshold

    # In spectrum, peaks with intensity <= thr are considered noise and removed
    p = p[inten > thr]
    if p.shape[0] == 0:
        return p

    if max_peaks is not None and p.shape[0] > int(max_peaks):
        idx = np.argsort(p[:, 1])[::-1][: int(max_peaks)]
        p = p[idx]

    # Stability: sort by mz
    p = p[np.argsort(p[:, 0])]
    return p


def discard_low_quality(peaks: np.ndarray, min_peaks: int) -> Optional[np.ndarray]:
    """
    If remaining peaks < min_peaks, discard the spectrum (return None).
    """
    p = _to_array(peaks)
    if p.shape[0] < int(min_peaks):
        return None
    return p


def scale_to_unit_norm(peaks: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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
    steps: list[dict],
    precursor_mz: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Apply the list of preprocessing functions in order.
    Each step is a dict with 'function' and 'params' keys.
    
    Example:
        steps = [
            {'function': 'set_mz_range', 'params': {'min_mz': 100.0, 'max_mz': 2000.0}},
            {'function': 'remove_precursor_peak', 'params': {'tol': 1.5, 'unit': 'Da'}},
        ]
    """
    x: Optional[np.ndarray] = peaks
    
    for step in steps:
        if x is None:
            return None
            
        func_name = step['function']
        params = step.get('params', {})
        
        if func_name == 'set_mz_range':
            x = set_mz_range(x, **params)
        elif func_name == 'remove_precursor_peak':
            x = remove_precursor_peak(x, precursor_mz=precursor_mz, **params)
        elif func_name == 'scale_intensity':
            x = scale_intensity(x, **params)
        elif func_name == 'filter_intensity':
            x = filter_intensity(x, **params)
        elif func_name == 'discard_low_quality':
            x = discard_low_quality(x, **params)
        elif func_name == 'scale_to_unit_norm':
            x = scale_to_unit_norm(x, **params)
        else:
            raise ValueError(f"Unknown function: {func_name}")
    
    return x


import os
import numpy as np
from .mgf_parse import load_spectra_npz, save_spectra_npz

IN_NPZ  = "output/spectra_raw.npz"
OUT_NPZ = "output/spectra_filtered.npz"


MIN_MZ = 100.0
MAX_MZ = 2000.0
MIN_INTENSITY = 1e-12    
MAX_PEAKS = 200
MIN_PEAKS = 5
REMOVE_PRECURSOR_TOL = 1.5
PRECURSOR_UNIT = "Da"  

def main():

    spectra = load_spectra_npz(IN_NPZ)
    print(f"[peak_filter] loaded spectra: {len(spectra)} from {IN_NPZ}")

    # Output the first 100 lines to a txt file
    out_txt = "output/spectra_first100.txt"
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for i, sp in enumerate(spectra[:100]):
            f.write(f"# Spectrum {i+1}\n")
            f.write(str(sp) + "\n\n")
    print(f"[peak_filter] wrote first 100 spectra to {out_txt}")


    # Automatically calculate parameters
    all_mz = np.concatenate([np.array(sp["peaks"], dtype=float)[:,0] for sp in spectra if len(sp["peaks"]) > 0])
    all_intensity = np.concatenate([np.array(sp["peaks"], dtype=float)[:,1] for sp in spectra if len(sp["peaks"]) > 0])
    peak_counts = np.array([len(sp["peaks"]) for sp in spectra])

    MIN_MZ = float(np.min(all_mz))
    MAX_MZ = float(np.max(all_mz))
    MIN_INTENSITY = 0.01   # Retain peaks with intensity > 1% of max peak
    MAX_PEAKS = int(np.median(peak_counts))
    MIN_PEAKS = int(np.min(peak_counts))
    REMOVE_PRECURSOR_TOL = 1.5
    PRECURSOR_UNIT = "Da"

    print(f"[peak_filter] auto params: MIN_MZ={MIN_MZ}, MAX_MZ={MAX_MZ}, MIN_INTENSITY={MIN_INTENSITY}, MAX_PEAKS={MAX_PEAKS}, MIN_PEAKS={MIN_PEAKS}")

    pipeline = [
        {'function': 'set_mz_range', 'params': {'min_mz': MIN_MZ, 'max_mz': MAX_MZ}},
        {'function': 'remove_precursor_peak', 'params': {'tol': REMOVE_PRECURSOR_TOL, 'unit': PRECURSOR_UNIT}},
        {'function': 'scale_intensity', 'params': {'scaling': 'root', 'max_intensity': 1}},
        {'function': 'filter_intensity', 'params': {'min_intensity': MIN_INTENSITY, 'max_peaks': MAX_PEAKS}},
        {'function': 'discard_low_quality', 'params': {'min_peaks': MIN_PEAKS}},
        {'function': 'scale_to_unit_norm', 'params': {}},
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

        # Save back to the same Spectrum format (preferably followed by binning/SSL)
        sp2 = {
            "meta": sp["meta"],
            "peaks": list(map(tuple, out.tolist())),  # list[(mz,inten)]
        }
        filtered.append(sp2)


    print(f"[peak_filter] kept={len(filtered)} dropped={dropped}")

    # Output the spectrograms after the first 100 filters to a txt file
    out_txt2 = "output/spectra_filtered_first100.txt"
    os.makedirs(os.path.dirname(out_txt2) or ".", exist_ok=True)
    with open(out_txt2, "w", encoding="utf-8") as f:
        for i, sp in enumerate(filtered[:100]):
            f.write(f"# Filtered Spectrum {i+1}\n")
            f.write(str(sp) + "\n\n")
    print(f"[peak_filter] wrote first 100 filtered spectra to {out_txt2}")

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
