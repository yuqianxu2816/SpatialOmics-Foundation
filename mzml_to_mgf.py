from pyteomics import mzml, mgf

spectra = []
print("hii")
with mzml.read("09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic.mzML") as reader:
    for sp in reader:
        print("hi")
        if sp.get("ms level") == 2:
            spectra.append({
                "m/z array": sp["m/z array"],
                "intensity array": sp["intensity array"],
                "params": {
                    "pepmass": sp["precursorList"]["precursor"][0]
                                ["selectedIonList"]["selectedIon"][0]
                                ["selected ion m/z"]
                }
            })

mgf.write(spectra, "output.mgf")
