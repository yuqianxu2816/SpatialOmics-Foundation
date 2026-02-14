import math
import numpy as np
import pytest

from SpatialOmics_Foundation import mgf_parse as m

def _write_tiny_mgf(path):
    # BEGIN/END blocks; first has bad lines; second misses meta data
    content = """BEGIN IONS
PEPMASS=512.34 9999
CHARGE=2+
RTINSECONDS=123.4
100.0 10.0
200.0 0.0
bad_line_should_be_ignored
END IONS

BEGIN IONS
PEPMASS=700.0
# missing CHARGE and RTINSECONDS
150.0 5.5
END IONS
"""
    path.write_text(content, encoding="utf-8")


def test_parse_mgf_spectrum_count(tmp_path):
    mgf = tmp_path / "tiny.mgf"
    _write_tiny_mgf(mgf)

    spectra = m.parse_mgf(str(mgf))
    assert isinstance(spectra, list)
    assert len(spectra) == 2  # Test spectrum count 


def test_parse_mgf_peaks_validity_and_nonempty(tmp_path):
    mgf = tmp_path / "tiny.mgf"
    _write_tiny_mgf(mgf)
    spectra = m.parse_mgf(str(mgf))

    for sp in spectra:
        peaks = sp["peaks"]
        assert len(peaks) > 0  # peaks should not be empty
        for mz, inten in peaks:
            assert mz > 0
            assert inten >= 0


def test_parse_mgf_metadata_fields_and_missing_branch(tmp_path):
    mgf = tmp_path / "tiny.mgf"
    _write_tiny_mgf(mgf)
    spectra = m.parse_mgf(str(mgf))

    sp0 = spectra[0]["meta"]
    assert sp0["CHARGE"] == 2
    assert abs(sp0["PEPMASS"] - 512.34) < 1e-6
    assert abs(sp0["RTINSECONDS"] - 123.4) < 1e-6

    sp1 = spectra[1]["meta"]
    # meta data missing -> return none
    assert sp1["CHARGE"] is None
    assert sp1["RTINSECONDS"] is None
    assert abs(sp1["PEPMASS"] - 700.0) < 1e-6


@pytest.mark.parametrize(
    "charge_str, expected",
    [
        ("2+", 2),
        ("2", 2),
        ("2+ and 3+", 2),   # Multiple charges -> take the first token
        ("", None),
        (None, None),
        ("abc", None),
    ],
)
def test_parse_charge(charge_str, expected):
    assert m._parse_charge(charge_str) == expected


@pytest.mark.parametrize(
    "pepmass_str, expected",
    [
        ("512.34", 512.34),
        ("512.34 9999", 512.34),
        ("", None),
        (None, None),
        ("abc", None),
    ],
)
def test_parse_pepmass(pepmass_str, expected):
    out = m._parse_pepmass(pepmass_str)
    if expected is None:
        assert out is None
    else:
        assert abs(out - expected) < 1e-9


def test_save_and_load_npz_roundtrip(tmp_path):
    mgf = tmp_path / "tiny.mgf"
    _write_tiny_mgf(mgf)
    spectra = m.parse_mgf(str(mgf))

    out_npz = tmp_path / "out.npz"
    m.save_spectra_npz(spectra, str(out_npz))
    assert out_npz.exists()

    spectra2 = m.load_spectra_npz(str(out_npz))
    assert len(spectra2) == len(spectra)

    # Key fields should match
    assert spectra2[0]["meta"]["CHARGE"] == 2
    assert abs(spectra2[0]["meta"]["PEPMASS"] - 512.34) < 1e-6

    # peaks length should match
    assert len(spectra2[0]["peaks"]) == len(spectra[0]["peaks"])
